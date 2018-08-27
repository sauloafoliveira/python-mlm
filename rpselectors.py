import numpy as np

class Selector:

    def __init__(self):
        pass

    def select(self, X, y):
        return np.ones((len(X), 1)), X, y


class RandomSelector(Selector):

    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        if self.k is None:
            self.k = round(np.log10(len(X)) * 5).astype('int')

        perm = np.random.permutation(X.shape[0])
        perm = perm[:self.k]

        if len(y.shape) == 1:
            return perm, X[perm], y[perm]

        return perm, X[perm], y[perm]


class CondensedSelector(Selector):
    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier

        if self.k is None:
            self.k = round(5 * np.log10(len(X)))

        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X, y)

        out = knn.predict(X)
        idx = out != y
        return idx, X[idx], y[idx]


class EditedSelection(Selector):
    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier

        if self.k is None:
            self.k = round(5 * np.log10(len(X)))

        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X, y)

        out = knn.predict(X)
        idx = out == y
        return idx, X[idx], y[idx]


class FastSelector(Selector):
    def __init__(self, k=16, p=0.75, distance_threshold=None):
        self.k = k
        self.p = 0.55
        self.distance_threshold = distance_threshold

    def _candidates(self, X, y):
        self.nn.fit(X, y)

        dist, yhat = self.nn.radius_neighbors(X)

        result = [self._iscorner(X[i], y[i], y[yhat[i]]) for i in range(len(X))]

        corner, score = zip(*result)
        corner = np.array(corner)
        score = np.array(score)

        return X[np.array(corner)], y[corner], score[corner]


    def _nonmax_supress(self, Xcand, ycand, corner_response):

        perm = np.random.permutation(len(Xcand))
        Xcand = Xcand[perm]
        ycand = ycand[perm]
        corner_response = corner_response[perm]

        self.nn.fit(Xcand, ycand)
        dist, ind = self.nn.radius_neighbors()

        idx = list(filter(lambda i: np.all(corner_response[i] > corner_response[ind[i]]), range(len(Xcand))))
        idx = np.array(idx)
        return Xcand[idx], ycand[idx]

    def _iscorner(self, x, y, y_neighbors):
        from scipy.spatial.distance import cdist

        if np.isscalar(y):
            y = np.array([y])
            y = y[:, None]
            y_neighbors = y_neighbors[:, None]

        if len(y_neighbors) == self.k and self._allsame(y_neighbors):
            return not(np.all(y == y[0])), self.k
        else:
            d = cdist(y, y_neighbors)
            score = self.k - np.sum(d[d > 0])
            return (score / self.k) > self.p, score


    def _allsame(self, y):
        from scipy.spatial.distance import cdist
        if len(y) == 0:
            return True
        return np.all(y == y[0])

    def select(self, X, y):
        if self.distance_threshold is None:
            from sklearn.neighbors import KNeighborsRegressor

            nn = KNeighborsRegressor(n_neighbors=self.k + 1)
            nn.fit(X, y)
            dist, ind = nn.kneighbors(X)

            self.distance_threshold = np.max(np.min(dist[:, 1:], 1))

        from sklearn.neighbors import RadiusNeighborsRegressor
        self.nn = RadiusNeighborsRegressor(radius=self.distance_threshold)

        Xcand, ycand, corner_response = self._candidates(X, y)

        Xcorner, ycorner = self._nonmax_supress(Xcand, ycand, corner_response)

        idx = np.where(np.isin(X, Xcorner))
        return idx, Xcorner, ycorner