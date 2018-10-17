import numpy as np

from sklearn.preprocessing import StandardScaler
from mlm import MinimalLearningMachine as MLM
from mlm.selectors import KSSelection, NLSelection
from sklearn.utils.validation import check_X_y

import matplotlib.pyplot as plt


mydata = np.genfromtxt('/Users/sauloafoliveira/Dropbox/thesis_code/mcycle.csv', delimiter=",")

X = mydata[:, 0].reshape(-1, 1)
y = mydata[:, 1].reshape(-1, 1)

X, y = check_X_y(X, y)


scaler = StandardScaler().fit(X)
X = scaler.transform(X)

mlm1 = MLM(selector=KSSelection())
mlm1.fit(X, y)
r = mlm1.score(X, y)



mlm2 = MLM(selector=NLSelection())
mlm2.fit(X, y)
s = mlm2.score(X, y)

print(r, s)
print(mlm1.sparsity(), mlm2.sparsity())


f, ax = plt.subplots(2, 2, sharey=True)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

svrr = SVR(C=128).fit(X, y)
ytrue = svrr.predict(X)


knnr1 = KNeighborsRegressor().fit(X, y)
knnr2 = KNeighborsRegressor().fit(mlm2.M, mlm2.t)

ax = ax.ravel()

cl = [mlm1, mlm2,  knnr1, knnr2 ]
for i in range(len(ax)):

    ax[i].plot(X, y, '.r', alpha=0.5)
    ax[i].plot(X, ytrue, '-k', alpha=0.8)
    ax[i].plot(X, cl[i].predict(X), '-b')
    #ax[i].plot(mlm1.M, mlm1.t, '.k')
    ax[i].set_title('Model: {}'.format(round(cl[i].score(X, y), 2)))
#
# ax2.plot(X, y, '.r', alpha=0.5)
# ax2.plot(X, mlm2.predict(X), '-b')
# ax2.plot(mlm2.M, mlm2.t, '.k')
# ax2.set_title('NLSelection {} -> {}'.format(round(s, 2), mlm2.sparsity()))
#
#
# ax3.plot(X, y, '.r', alpha=0.5)
# ax3.plot(X, knnr.predict(X), '-b')
# ax3.plot(mlm2.M, mlm2.t, '.k')
# ax3.set_title('KNN for R {} -> {}'.format(round(knnr.score(X, y), 2), mlm2.sparsity()))
#
#



plt.show()
