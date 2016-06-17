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

def calc_rmse(truth, estimate):
	return np.sqrt(((estimate - truth) ** 2).mean())



data = np.loadtxt("../dataSets/linRegData.txt")
trainsize = 20
var = 0.02

x_test = data[trainsize:,0]
x_train = data[0:trainsize,0]

y_test = data[trainsize:,1]
y_train = data[0:trainsize,1]

rmse_x = []
rmse_y = []

# calc curves and plot them
for degree in range(15, 40):
	space = np.linspace(0, 2, num=degree)
	phi = normalize(get_phi_gaussians(space, x_train, var))
	w = np.dot(np.dot(np.linalg.inv(np.dot(phi, phi.transpose()) + math.exp(-6)*np.eye(len(phi))), phi), y_train)
	X = get_phi_gaussians(space, x_test, var)


	# fit curve
	ye_test = np.dot(w, X).transpose()
	rmse = calc_rmse(y_test, ye_test)
	rmse_x.append(degree)
	rmse_y.append(rmse)


plt.plot(rmse_x, rmse_y)
#plt.yscale('log')
plt.show()

