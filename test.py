from sklearn import datasets
from rpselectors import FastSelector
from sklearn.preprocessing import StandardScaler






boston = datasets.load_boston()

X = boston.data
y = boston.target

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

from mlm import CubicMinimalLearningMachine as CBMLM, MinimalLearningMachine as MLM
#
# cb = CBMLM()
# cb.fit(X, y)
# print(cb.score(X, y))

mlm = MLM()
mlm.fit(X, y)
print(mlm.score(X, y))



iris = datasets.load_iris()
X = iris.data
y = iris.target

X = scaler.fit(X).transform(X)


from rpselectors import FastSelector

f = FastSelector()


from mlm import NearestNeighborMinimalLearningMachineClassifier as NNMLM, MinimalLearningMachineClassifier as MLMC


mlmc = MLMC()
mlm.fit(X, y)
print(mlm.score(X, y))



mlmc = MLMC(selector=f)
mlm.fit(X, y)
print(mlm.score(X, y))
