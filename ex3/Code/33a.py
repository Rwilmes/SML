import matplotlib.pyplot as plt
import numpy as np


def plotArray(X, c):
    for i in np.arange(0, X.shape[0]):
        plt.plot(X[i, 1], X[i, 2], c)

    return


def normalize(X):
    # Subtract mean
    m = np.array([np.mean(X, 0)])
    m = np.repeat(m, X.shape[0], 0)
    X = np.subtract(X, m)

    # Devide by standard deviation
    dev = np.array([np.std(X, 0)])
    X = np.divide(X, dev)

    return X, m, dev


def pca(X):
    covar = np.cov(X, rowvar=False)
    [lambdas, evec] = np.linalg.eig(covar)
    return lambdas, evec


def unnormalize(X, m, dev):
    # multiply with standard deviation
    X = np.multiply(X, dev)

    # add mean
    X = np.add(X, m)

    return X


# Load data and split into features and classes
data = np.loadtxt('../dataSets/iris.txt', delimiter=',')
classes = data[:, 4]
data = data[:, 0:4]

# Separate samples of different classes
c0 = np.equal(classes, 0)
c1 = np.equal(classes, 1)
c2 = np.equal(classes, 2)

# Normalize data and perform PCA
[data, m, dev] = normalize(data)
[eval, evec] = pca(data)

# Calculate explained variance
explainedVariance = np.cumsum(eval) / np.sum(eval)
print "Explained: ", explainedVariance

# Calculate number of features necessary to explain at least 95% variance
componentsNeeded = 0
while explainedVariance[componentsNeeded] < 0.95:
    componentsNeeded += 1

# Project data onto principal components
datap = np.dot(evec.T, data.T).T

# Plot result
plt.plot(datap[c0, 0], datap[c0, 1], 'o', label='Setosa')
plt.plot(datap[c1, 0], datap[c1, 1], 'o', label='Versicolour')
plt.plot(datap[c2, 0], datap[c2, 1], 'o', label='Virginica')
plt.xlabel("Eigenvactor 1")
plt.ylabel('Eigenvecotr 2')
plt.legend(loc='best')
plt.show()

for i in np.arange(1, 5):
    # Set all unused principal components to 0
    data_restored = np.copy(datap)
    data_restored[:, i:] = 0

    # Project features back to original feature space
    data_restored = np.linalg.solve(evec.T, data_restored.T).T

    # Calculate RMSE
    e = np.subtract(data, data_restored)
    e **= 2
    e = np.sum(e, axis=1)
    e = np.divide(np.sum(e, axis=0), e.size)
    e = np.sqrt(e)
    print e
