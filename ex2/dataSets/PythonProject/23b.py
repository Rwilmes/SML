import matplotlib.pyplot as plt
import numpy as np


def calculateDensity(evalat, sample, sigma):
    nelements = sample.size
    nsteps = evalat.size
    #repeat vectors to facilitate calculation
    evalat = evalat.repeat(nelements, 0)
    sample = sample.repeat(nsteps, 1)

    # calculate exponent
    p = np.negative(np.power(np.abs(np.subtract(evalat, sample)), 2))
    p = np.divide(p, 2 * sigma * sigma)
    # exponantiate and sum up
    p = np.exp(p)
    p = np.sum(p, 0)
    # divide
    p = np.divide(p, nelements * np.sqrt(2 * np.pi) * sigma)

    return p


# Load the dataset
samples = np.array([np.loadtxt(fname='../nonParamTrain.txt')]).transpose()
testset = np.array([np.loadtxt(fname='../nonParamTest.txt')]).transpose()
x = np.array([np.arange(-4, 8, 0.1)])

for sigma in [0.03, 0.2, 0.8]:
    distribution = calculateDensity(x, samples, sigma)

    # calculate likelihood
    likelihood = calculateDensity(samples.transpose(), samples, sigma)
    likelihood = np.sum(np.log(likelihood))

    # Calculate likelihood of Testset
    likelihood_t = calculateDensity(testset.transpose(), samples, sigma)
    likelihood_t = np.sum(np.log(likelihood_t))
    print likelihood_t

    # plot result
    plt.plot(x[0, ...], distribution, label="KDE with sigma {} and likelihood {}".format(sigma, likelihood))
    plt.title("KDE")

plt.legend(fontsize="small", loc="best")

plt.show()
