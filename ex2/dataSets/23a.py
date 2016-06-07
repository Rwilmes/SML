import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(fname='nonParamTrain.txt')

f, axarr = plt.subplots(3, sharex=True)

n = 0

for width in [0.02, 0.5, 2.0]:
    boundaries = np.arange(data.min(), data.max(), width)

    axarr[n].hist(data, bins=boundaries)
    axarr[n].set_title("Histogram with bins of width")
    #axarr[n].xlabel("Value")
    #axarr[n].ylabel("Frequency")

    f.show()

    n += 1

plt.show()