import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

# normalizes all ys over x
def normalize(ys, x):
	for i in range(0, len(x)):
		total = 0
		for y in ys:
			total += y[i]

		for y in ys:
			y[i] = y[i]/total

	return ys

def get_gaussians(space, x, var):
	ys = []
	for i in space:
		mu = i
		sigma = math.sqrt(var)	
		y = mlab.normpdf(x, mu, sigma)
		ys.append(y)
	return ys

space = np.linspace(0, 2, num=20)
x = np.linspace(0, 2, 1000)

ys = get_gaussians(space, x, 0.02)
ys = normalize(ys, x)

for y in ys:
	plt.plot(x,y)

plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()
