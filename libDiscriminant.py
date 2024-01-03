# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# list of points
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(22)

means = [[0, 5], [5, 0]]
cov0 = [[4, 3], [3, 4]]
cov1 = [[3, 1], [1, 1]]
N0 = 50
N1 = 40
N = N0 + N1
X0 = np.random.multivariate_normal(means[0], cov0, N0) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov1, N1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.concatenate((X0, X1))
y = np.array([0]*N0 + [1]*N1)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

W = clf.coef_/np.linalg.norm(clf.coef_) # normalize
#w = W[:,0]
plt.figure(figsize=(10,8))

# Biểu diễn X0
plt.scatter(X0[:, 0], X0[:, 1], color='b', alpha=0.5, label='N1')

# Biểu diễn X1
plt.scatter(X1[:, 0], X1[:, 1], color='r', alpha=0.5, label='N2')
x = np.linspace(-5, 5, 100)

# Tính toán các giá trị tương ứng cho trục y
y = W[0][0]/W[0][1] * x

# Vẽ đường thẳng
plt.plot(x, y, label='LDA')
plt.legend()
plt.show()