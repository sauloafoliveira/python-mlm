import numpy as np

__author__ = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__ = "production"
__version__ = "1.0.0"
__date__ = "07 September 2018"


class SelectionAlgorithm:
    def __init__(self):
        pass

    def select(self, X, y):
        return np.ones((len(X), 1), bool), X, y


class RandomSelection(SelectionAlgorithm):
    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        if self.k is None:
            self.k = round(np.log10(len(X)) * 5).astype('int')

        perm = np.random.permutation(len(X))
        perm = perm[:self.k]

        if len(y.shape) == 1:
            return perm, X[perm], y[perm]

        idx = np.zeros((len(X), 1), bool)
        idx[perm] = True

        return perm, X[perm], y[perm]


class CondensedSelection(SelectionAlgorithm):
    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier

        if self.k is None:
            self.k = round(5 * np.log10(len(X)))

        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X, y)

        out = knn.predict(X)
        idx = (out != y)
        return idx, X[idx], y[idx]


class EditedSelection(SelectionAlgorithm):
    def __init__(self, k=None):
        self.k = k

    def select(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier

        if self.k is None:
            self.k = round(5 * np.log10(len(X)))

        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X, y)

        out = knn.predict(X)
        idx = (out == y)
        return idx, X[idx], y[idx]


class FASTSelection(SelectionAlgorithm):
    def __init__(self, k=16, p=0.75, distance_threshold=None):
        self.k = k
        self.p = p
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

        idx = list(filter(lambda i: np.all(corner_response[i] >= corner_response[ind[i]]), range(len(Xcand))))
        idx = np.array(idx)
        return Xcand[idx], ycand[idx]

    def _iscorner(self, x, y, y_neighbors):
        from scipy.spatial.distance import cdist

        if np.isscalar(y):
            y = np.array([y])
            y = y[:, None]
            y_neighbors = y_neighbors[:, None]

        if (cdist(y, y_neighbors[0]) > 0) and self._allsame(y_neighbors):
            return True, self.k
        else:
            d = cdist(y, y_neighbors)
            score = self.k - np.sum(d[d > 0])
            return (score / self.k) > self.p, score

    @staticmethod
    def _allsame(y):
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


class AcitveSelection(SelectionAlgorithm):
    '''
    Optimized Fixed-Size Kernel Models for Large Data Sets
    '''

    def __init__(self, M=None, l=1, trials=None):
        self.l = l
        self.M = M
        self.trials = trials

    def select(self, X, y):
        idx, Xr, yr = RandomSelection(k=self.M).select(X, y)

        old_crit = np.Inf

        for trial in range(1, self.trials + 1):
            old_Xr = Xr
            old_yr = yr

            i = np.random.randint(0, self.M)
            j = np.random.randint(0, len(X))

            Xr[i] = X[j]
            yr[i] = y[j]

            crit = self.__quad_renyi_entropy(Xr, self.l)

            if old_crit <= crit:
                # undo permutation
                Xr = old_Xr
                yr = old_yr

        idx = np.where(np.isin(X, Xr))
        return idx, Xr, yr

    @staticmethod
    def __quad_renyi_entropy(X, l):

        from scipy.spatial.distance import cdist

        Dx = cdist(X, X)

        np.fill_diagonal(Dx, l)

        U, lam = np.linalg.eig(Dx)

        if lam.shape[0] == lam.shape[1]:
            lam = np.diagonal(lam)

        return np.log(np.power(np.mean(U), 2) * lam)


class CriticalSelection(SelectionAlgorithm):
    def __init__(self, kb=None, l=0.3, ke=None, gamma=0.1):
        self.kb = kb
        self.ke = ke
        self.l = l
        self.gamma = gamma

    def select(self, X, y):
        from scipy.stats import mode

        from sklearn.neighbors import KNeighborsClassifier

        if self.kb is None:
            self.kb = round(5 * np.log10(len(X)))

        knn = KNeighborsClassifier(n_neighbors=self.kb + 1)
        knn.fit(X, y)

        kneighbors = knn.kneighbors(X)

        w, freq = mode(kneighbors[0])

        # 1st test

        r = (y == w)

        test1 = (freq[r] > 1) and (np.power(freq[r], -1) <= w[r]) and (w[r] <= np.power(freq[r], -1) + self.l)

        idx = np.zeros((1, len(X)), bool)
        idx[test1] = True

        # 2nd test
        test2 = w[r] > (np.power(len(np.unique(y)), -1) + self.l)

        possible_edges = zip(X[not test1 and test2], y[not test1 and test2])

        actual_edges = [self.edge_pattern_selection(xy, X) for xy in possible_edges]

        idx[not test1 and test2] = actual_edges

        return idx, X[idx], y[idx]

    def edge_pattern_selection(self, xy, X):
        from scipy.stats import zscore

        V = X - xy[0]

        Z = zscore(V)

        Vin = np.sum(Z)

        theta = Vin @ V

        l = np.sum(theta >= 0) / len(X)

        return l >= (1 - xy[1])


class NLSelection(SelectionAlgorithm):

    def __init__(self, cutoff=(.2, .32), k=None):
        self.cutoff = cutoff
        self.k = k

    @staticmethod
    def __moving_average(a, periods):
        weights = np.ones(1, periods) / periods
        return np.convolve(a, weights, mode='valid')

    @staticmethod
    def __order_of(X):
        from scipy.spatial.distance import cdist

        x_origin = np.min(X, axis=0)
        keys = cdist(np.asmatrix(x_origin), X)

        return np.argsort(keys).T


    def select(self, X, y):
        from scipy.signal import find_peaks

        n = len(X)

        # trick to sort rows
        order = self.__class__.__order_of(X)

        X, y = X[order], y[order]

        yl = y.reshape(-1, ) # self.__moving_average(y, X.shape[-1] + 1)

        s = round(np.sqrt(np.nanstd(yl)))

        h_peaks, l = find_peaks(yl, distance=s)
        l_peaks, l = find_peaks(-yl, distance=s)

        idx = np.zeros((n, 1), dtype=bool)

        idx[h_peaks] = True
        idx[l_peaks] = True

        idx = np.asmatrix(idx)

        return idx, X[idx], y[idx]


class KSSelection(SelectionAlgorithm):

    def __init__(self, cutoff=(.2, .32), k=None):
        self.cutoff = cutoff
        self.k = k

    @staticmethod
    def __pval_ks_2smap(entry):
        from scipy.stats import ks_2samp, zscore

        a = zscore(entry[0]) if np.std(entry[0]) > 0 else entry[0]

        b = zscore(entry[1]) if np.std(entry[1]) > 0 else entry[1]

        info, pval = ks_2samp(a, b)

        return pval


    def select(self, X, y):
        from sklearn.neighbors import NearestNeighbors

        n = len(X)

        if self.k is None:
            self.k = round(5 * np.log10(n))

        knn = NearestNeighbors(n_neighbors=int(self.k + 1), algorithm='ball_tree').fit(X)

        distx, ind = knn.kneighbors(X)

        knn = NearestNeighbors(n_neighbors=int(self.k + 1), algorithm='ball_tree').fit(y)

        disty, ind = knn.kneighbors(y, return_distance=True)

        zipped = list(zip(distx[:, :1], disty[:, :1]))

        p = [self.__pval_ks_2smap(entry) for entry in zipped]

        order = np.argsort(p)

        h_cutoff = round(self.cutoff[0] * n)
        l_cutoff = round(self.cutoff[1] * n)

        idx = np.zeros((n, 1), dtype=bool)
        idx[order[:l_cutoff]] = True
        idx[order[-h_cutoff:]] = True

        idx = idx.reshape(-1, )

        return idx, X[idx], y[idx]


