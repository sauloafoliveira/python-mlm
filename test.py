from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from mlm import MinimalLearningMachine as MLM
from mlm import MinimalLearningMachineClassifier as MLMC

import unittest

class TestStringMethods(unittest.TestCase):


    def test_iris(self):

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        scaler = StandardScaler().fit(X)

        mlm = MLMC()
        mlm.fit(X, y)
        acc = mlm.score(X, y)

        self.assertGreater(acc, 0, 'R2 {}'.format(acc))

    def test_boston(self):

        boston = datasets.load_boston()

        X = boston.data
        y = boston.target

        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        mlm = MLM()
        mlm.fit(X, y)
        mlm_r2 = mlm.score(X, y)

        self.assertGreater(mlm_r2, 0, 'R2 {}'.format(mlm_r2))


if __name__ == '__main__':
    unittest.main()

