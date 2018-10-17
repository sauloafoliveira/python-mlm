from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from mlm import MinimalLearningMachine as MLM
from mlm.selectors import KSSelection, NLSelection

boston = datasets.load_boston()

X = boston.data
y = boston.target

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

mlm1 = MLM()
mlm1.fit(X, y)
mlm_r1 = (mlm1.score(X, y))

mlm2 = MLM(selector=KSSelection())
mlm2.fit(X, y)
mlm_r2 = mlm2.score(X, y)

mlm3 = MLM(selector=NLSelection())
mlm3.fit(X, y)
mlm_r3 = mlm3.score(X, y)

print(f'RN: R2 of {round(mlm_r1, 2)} with sparsity of {mlm1.sparsity()}')
print(f'KS: R2 of {round(mlm_r2, 2)} with sparsity of {mlm2.sparsity()}')
print(f'NL: R2 of {round(mlm_r3, 2)} with sparsity of {mlm3.sparsity()}')


print('*' * 30)

boston = datasets.load_boston()

X = scaler.transform(boston.data)
y = boston.target


mlm1 = MLM()
mlm1.fit(X, y)
mlm_r1 = (mlm1.score(X, y))

mlm2 = MLM(selector=KSSelection())
mlm2.fit(X, y)
mlm_r2 = mlm2.score(X, y)

mlm3 = MLM(selector=NLSelection())
mlm3.fit(X, y)
mlm_r3 = mlm3.score(X, y)

print(f'RN: R2 of {round(mlm_r1, 2)} with sparsity of {mlm1.sparsity()}')
print(f'KS: R2 of {round(mlm_r2, 2)} with sparsity of {mlm2.sparsity()}')
print(f'NL: R2 of {round(mlm_r3, 2)} with sparsity of {mlm3.sparsity()}')

print('*' * 30)
