from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.optimize import root
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelBinarizer
from mlm.selectors import RandomSelection


__all__ = ['MinimalLearningMachine', 'MinimalLearningMachineClassifier',
           'NearestNeighborMinimalLearningMachineClassifier',
           'CubicMinimalLearningMachine']

__author__  = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__  = "production"
__version__ = "1.0.0"
__date__    = "07 September 2018"



class MinimalLearningMachine(BaseEstimator, RegressorMixin):

    def __init__(self, selector=None, estimator_type='regressor'):
        self.selector = RandomSelection() if selector is None else selector
        self.M = []
        self.t = []

    def fit(self, X, y):

        if len(y.shape) == 1:
            y = y[:, None]

        idx, self.M, self.t = self.selector.select(X, y)

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        dx = cdist(X, self.M)
        dy = cdist(y, self.t)

        self.B_ = np.linalg.pinv(dx) @ dy

        return self

    def mulat_(self, y, dyh):
        return np.sum(np.power(np.power(cdist(np.asmatrix(y), self.t), 2) - np.power(dyh, 2), 2))

    def active_(self, dyhat):
        y0h = np.mean(self.t)

        result = [root(method='lm', fun=lambda y: self.mulat_(y, dyh), x0=y0h) for dyh in dyhat]
        yhat = list(map(lambda y: y.x, result))
        return np.asmatrix(yhat)

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        dyhat = cdist(X, self.M) @ self.B_

        return self.active_(dyhat)



class MinimalLearningMachineClassifier(MinimalLearningMachine, ClassifierMixin):

    def __init__(self, selector=None):
        MinimalLearningMachine.__init__(self, selector)
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


