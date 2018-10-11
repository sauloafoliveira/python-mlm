from metric_learn import LMNN, ITML_Supervised, LSML_Supervised, SDML_Supervised
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from mlm import MinimalLearningMachine as MLM
from mlm import MinimalLearningMachineClassifier as MLMC
from mlm import NearestNeighborMinimalLearningMachineClassifier as NNMLMC
from mlm.selectors import KSSelection, NLSelection

db = datasets.load_breast_cancer()
X = db.data
y = db.target

scaler = StandardScaler().fit(X)
X = scaler.transform(X)


mlm1 = NNMLMC()
mlm1.fit(X, y)
mlm_r1 = (mlm1.score(X, y))


mlm2 = NNMLMC(selector=KSSelection())
mlm2.fit(X, y)
mlm_r2 = mlm2.score(X, y)


mlm3 = NNMLMC(selector=NLSelection())
mlm3.fit(X, y)
mlm_r3 = mlm3.score(X, y)

print(f'RN: R2 of {round(mlm_r1, 2)} with sparsity of {round(1-len(mlm1.M)/len(X), 2)}')
print(f'KS: R2 of {round(mlm_r2, 2)} with sparsity of {round(1-len(mlm2.M)/len(X), 2)}')
print(f'NL: R2 of {round(mlm_r3, 2)} with sparsity of {round(1-len(mlm3.M)/len(X), 2)}')
