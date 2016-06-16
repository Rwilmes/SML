import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

def get_plot_points(x, y):
	points = {}
	for i in range(0, len(x_test)):
		points[x_test[i]]=y[i]
	x_sorted = []
	y_sorted = []
	for x in sorted(points.keys()):
		x_sorted.append(x)
		y_sorted.append(points[x])
	return x_sorted, y_sorted

def plot_phi(phi, x):
	for i in range(0, len(phi)):
		plt.plot(x, phi[i])
	plt.show()

def normalize(matrix):
	for i in range(0, len(matrix[0])):
		total = 0
		column = matrix[:,i]
		for j in range(0, len(column)):
			total += column[j]
		for j in range(0, len(column)):
			column[j] = column[j]/total
	return matrix

def get_phi_gaussians(space, x, var):
	matrix = np.zeros((len(space),len(x)))
	
	# fill matrix
	for i in range(0, len(matrix)):
		matrix[i] = x-space[i]
	matrix = np.exp(-np.power(matrix,2)/(2*var))
	return matrix

data = np.loadtxt("../dataSets/linRegData.txt")
trainsize = 20
var = 0.02

x_test = data[trainsize:,0]
x_train = data[0:trainsize,0]

y_test = data[trainsize:,1]
y_train = data[0:trainsize,1]

# plot data points
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')


# calc curves and plot them
for degree in range(15, 18):
	space = np.linspace(0, 2, num=degree)
	phi = normalize(get_phi_gaussians(space, x_train, var))
	w = np.dot(np.dot(np.linalg.inv(np.dot(phi, phi.transpose())), phi), y_train)
	X = get_phi_gaussians(space, x_test, var)


	# fit curve
	ye_test = np.dot(w, X).transpose()

	# sort points for linespoint plot
	x_sorted, y_sorted = get_plot_points(x_test, ye_test)

	# plot
	plt.plot(x_sorted, y_sorted)

plt.show()

