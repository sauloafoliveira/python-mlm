from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.optimize import root
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y
from mlm.selectors import RandomSelection


__all__ = ['MinimalLearningMachine',
           'MinimalLearningMachineClassifier',
           'NearestNeighborMinimalLearningMachineClassifier',
           'CubicMinimalLearningMachine']

__author__  = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__  = "production"
__version__ = "1.0.0"
__date__    = "07 September 2018"



class MinimalLearningMachine(BaseEstimator, RegressorMixin):

    def __init__(self, selector=None, estimator_type='regressor', bias=False, l=0):
        self.selector = RandomSelection(k=np.inf) if selector is None else selector
        self.M = []
        self.t = []
        self._sparsity_scores = (0, np.inf) # sparsity and norm frobenius
        self._estimator_type = estimator_type
        self.bias = bias
        self.l = l

    def __validade_params(self):
        if not(hasattr(self, 'bias')):
            self.bias


    def fit(self, X, y):
        from numpy.linalg import norm

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if y.ndim == 1:
            y = np.array([y.ravel()]).T

        idx, _, _ = self.selector.select(X, y)

        self.M = X[idx]
        self.t = y[idx]

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        dx = cdist(np.asmatrix(X), np.asmatrix(self.M))
        dy = cdist(np.asmatrix(y), np.asmatrix(self.t))

        # if self.bias:
        #     dx = np.concatenate(np.ones(len(X)), dx, axis=1)
        #
        # if self.l > 0 :
        #     dx2 = dx.T @ dx
        #     np.fill_diagonal(dx2, self.l + np.diagonal(dx2))
        #     self.B_ = np.linalg.pinv(dx2 @ dx.T) @ dy
        # else:

        self.B_ = np.linalg.pinv(dx) @ dy

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        return self

    def mulat_(self, y, dyh):
        if y.ndim == 1:
            y = np.array([y.ravel()]).T

        dy2t = cdist(np.asmatrix(y), np.asmatrix(self.t))
        return np.sum(np.power(np.power(dy2t, 2) - np.power(dyh, 2), 2))

    def active_(self, dyhat):
        y0h = np.mean(self.t)

        result = [root(method='lm', fun=lambda y: self.mulat_(y, dyh), x0=y0h) for dyh in dyhat]
        yhat = list(map(lambda y: y.x, result))
        return np.asarray(yhat)

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        X = np.asmatrix(X)

        dyhat = cdist(X, self.M) @ self.B_

        return self.active_(dyhat)

    def sparsity(self):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        s = np.round(self._sparsity_scores, 2)

        return s[0], s[1]


class MinimalLearningMachineClassifier(MinimalLearningMachine, ClassifierMixin):

    def __init__(self, selector=None):
        MinimalLearningMachine.__init__(self, selector, estimator_type='classifier')
        self.lb = LabelBinarizer()

    def fit(self, X, y=None):
        self.lb.fit(y)
        return MinimalLearningMachine.fit(self, X, self.lb.transform(y))

    def active_(self, dyhat):
        classes = self.lb.transform(self.lb.classes_)

        result = [np.argmin(list(map(lambda y_class: self.mulat_(y_class, dyh), classes))) for dyh in dyhat]

        return self.lb.inverse_transform(self.lb.classes_[result])

    def score(self, X, y, sample_weight=None):
        return ClassifierMixin.score(self, X, y, sample_weight)

class NearestNeighborMinimalLearningMachineClassifier(MinimalLearningMachineClassifier):

    def __init__(self, selector=None):
        MinimalLearningMachineClassifier.__init__(self, selector)

    def active_(self, dyhat):
        m = np.argmin(dyhat, 1)
        return self.t[m]

class CubicMinimalLearningMachine(MinimalLearningMachine):

    def active_(self, dyhat):
        a = len(self.t)
        b = -3 * np.sum(self.t)
        c = 3 * np.sum(np.power(self.t, 2)) - np.sum(np.power(dyhat, 2), 1)
        d = np.power(dyhat, 2) @ self.t - np.sum(np.power(self.t, 3))

        return [self.cases_(np.roots([a, b, c[i], d[i]]), dyhat[i]) for i in range(len(dyhat))]

    def cases_(self, roots, dyhat):
        r = list(map(np.isreal, roots))
        if np.sum(r) == 3:
            # Rescue the root with the lowest cost associated
            j = [self.mulat_(y, dyhat) for y in np.real(roots)]
            return np.real(roots[np.argmin(j)])
        else:
            # As True > False, then rescue the first real value
            return np.real(roots[np.argmax(r)])


