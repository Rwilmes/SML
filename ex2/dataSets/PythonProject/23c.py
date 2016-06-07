import matplotlib.pyplot as plt
import numpy as np


def calculateknn(evalat, samples, K):
    nelements = samples.size
    nsteps = evalat.size
    # repeat vectors to facilitate calculation
    evalat = evalat.repeat(nelements, 0)
    samples = samples.repeat(nsteps, 1)

    # Calculate distance from evaluation points to all samples
    distances = np.sort(np.abs(np.subtract(samples, evalat)), 0)

    # Get distances to K-nearest neighbour
    distances = distances[K-1, ...]

    # Calculate probability according to formula
    p = np.divide(K, np.multiply(distances, nelements*2) )

    return p

# Load the dataset
samples = np.array([np.loadtxt(fname='../nonParamTrain.txt')]).transpose()
testset = np.array([np.loadtxt(fname='../nonParamTest.txt')]).transpose()
x = np.array([np.arange(-4, 8, 0.1)])

for K in [2, 8, 35]:
    distribution = calculateknn(x, samples, K)

    # calculate likelihood
    likelihood = calculateknn(samples.transpose(), samples, K)
    likelihood = np.sum(np.log(likelihood))

    # Calculate likelihood of Testset
    likelihood_t = calculateknn(testset.transpose(), samples, K)
    likelihood_t = np.sum(np.log(likelihood_t))
    print likelihood_t

    # plot result
    plt.plot(x[0, ...], distribution, label="KNN with K = {} and likelihood {}".format(K, likelihood))
    plt.title("KNN")

plt.legend(fontsize="small", loc="best")
plt.show()
