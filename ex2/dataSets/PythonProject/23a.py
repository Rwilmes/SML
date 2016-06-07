import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = np.loadtxt(fname='../nonParamTrain.txt')

# Create figure with 3 subplots
f, axarr = plt.subplots(3)

# Iterate through all bin-widths
n = 0
for width in [0.02, 0.5, 2.0]:
    # Calculate boundaries of bins
    boundaries = np.arange(data.min(), data.max(), width)

    # Plot histogram
    axarr[n].hist(data, bins=boundaries)
    axarr[n].set_title("Histogram with bins of width {}".format(width))
    n += 1

# Show plot
plt.show()
