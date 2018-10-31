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
        n = len(X)
        if self.k is None:
            k = round(np.log10(n) * 5).astype('int')
        elif np.isinf(self.k):
            k = len(X)
        elif np.isscalar(self.k):
            if self.k <= 1:
                k = np.round(self.k * len(X)).astype('int')
            else:
                k = np.round(max(min(self.k, len(X)), 2)).astype('int')

        perm = np.random.permutation(n)
        perm = perm[:k]

        idx = np.zeros(n, dtype=bool)

        idx[perm] = True

        idx = np.array(idx.ravel())

        return idx, X[idx], y[idx]


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

        return np.argsort(keys)


    def select(self, X, y):
        from scipy.signal import find_peaks

        n = len(X)

        # trick to sort rows
        order = self.__class__.__order_of(X)

        yl = y[order].ravel()

        s = np.round(np.sqrt(np.std(yl)))

        h_peaks, l = find_peaks(yl, distance=s)
        l_peaks, l = find_peaks(-yl, distance=s)

        idx = np.zeros(n, dtype=bool)

        idx[h_peaks] = True
        idx[l_peaks] = True

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

        _, pval = ks_2samp(a, b)

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

        idx = idx.ravel()

        return idx, X[idx], y[idx]


class DROP2_RE(SelectionAlgorithm):

    def __init__(self, k=None, a=0.1):
        from sklearn.neighbors import KNeighborsRegressor
        super(DROP2_RE, self).__init__()
        self.k = k
        self.alpha = a
        self.model = KNeighborsRegressor()

    def __err(self, Xy, associates, i):
        X, y = Xy

        self.model.fit(X[associates], y[associates])

        error_with = np.abs(self.model.predict(np.asmatrix(X[i])) - y[i])

        associates_without_x = list(set(associates) - set([i]))

        self.model.fit(X[associates_without_x], y[associates_without_x])

        error_without = np.abs(self.model.predict(np.asmatrix(X[i])) - y[i])

        return np.asarray([error_with, error_without])

    def __theta(self, y, A):

        associates = A[:self.k] if len(A) >= self.k else A

        return self.alpha * np.std( y[ associates ] )


    def select(self, X, y):
        from sklearn.neighbors import NearestNeighbors
        _, associates = NearestNeighbors(n_neighbors=len(X)).fit(X).kneighbors(X)

        Xy = (X, y)

        selected = np.ones(len(X), dtype=bool)

        for i in range(len(X)):
            A = associates[i]

            errors = np.zeros(2)

            for a in A:
                err = np.array(self.__err(Xy, A, a)).ravel()
                t = self.__theta(y, associates[a])
                errors += np.asarray(err < t)

            # error with <= error without
            if errors[0] <= errors[1]:
                selected[i] = False

                for a in A:
                    associates[a] = np.delete(associates[a], np.where(associates[a] == i))

        return selected, X[selected], y[selected]


class MutualInformationSelection(SelectionAlgorithm):

    def __init__(self, k=6, alpha=0.05):
        self.k = k
        self.alpha = alpha

    def select(self, X, y):
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neighbors import NearestNeighbors

        n = len(X)
        mask = np.arange(n)

        mi = [mutual_info_regression(X[mask != i], y[mask != i]) for i in range(n)]

        mi = MinMaxScaler().fit_transform(mi)

        _, neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(X).kneighbors(X)

        # dropout themselves
        neighbors = neighbors[:, 1:]

        cdiff = [np.sum((mi[i] - mi[neighbors[i]]) > self.alpha) for i in range(n)]

        idx = np.array(cdiff) < self.k

        return idx, X[idx], y[idx]


