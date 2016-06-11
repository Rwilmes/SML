import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

def normalize(y):
	total = 0
	for v in y:
		total += v	
	for i in range(0, len(y)):
		y[i] = float(y[i])/total
	return y

space = np.linspace(0, 2, num=20)

for i in space:
	mu = i
	variance = 0.02
	sigma = math.sqrt(variance)	
	x = np.linspace(0, 2, 100)
	y = mlab.normpdf(x, mu, sigma)
	y = normalize(y)	
	plt.plot(x,y)

plt.show()
