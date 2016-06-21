import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

# sorts the plot points
def get_plot_points(x, y):
	points = {}
	for i in range(0, len(x)):
		points[x[i]]=y[i]
	x_sorted = []
	y_sorted = []
	for x in sorted(points.keys()):
		x_sorted.append(x)
		y_sorted.append(points[x])
	return x_sorted, y_sorted

# normalizes rows
def normalize(matrix):
	for i in range(0, len(matrix)):
		matrix[i] = matrix[i]/np.sum(matrix[i])
	return matrix

# creates phi matrix with gaussians
def get_phi_gaussians(space, x, var):
	matrix = np.zeros((len(x), len(space)))
	for i in range(0, len(matrix[0])):
		matrix[:,i] = x
	for i in range(0, len(matrix)):
		matrix[i] = matrix[i] - space
	matrix = np.exp(-np.power(matrix, 2)/(2*var))
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

# plot data points
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')

rmse_x = []
rmse_y = []


# calc curves and plot them
for degree in range(15, 40):
	space = np.linspace(0, 2, num=degree)
	phi = normalize(get_phi_gaussians(space, x_train, var)).T

	w = phi.dot(phi.T)
	w = w + (10**(-6))*np.eye(len(w))
	w = np.linalg.inv(w).dot(phi).dot(y_train)

	X = normalize(get_phi_gaussians(space, x_test, var))

	# fit curve
	ye_test = X.dot(w).T

	# sort points for linespoint plot
	x_sorted, y_sorted = get_plot_points(x_test, ye_test)

	# plot
	plt.plot(x_sorted, y_sorted)

	# calc rmse
	rmse = calc_rmse(y_test, ye_test)
	rmse_x.append(degree)
	rmse_y.append(rmse)


plt.figure(2)
plt.ylabel('rmse')
plt.xlabel('number of basis functions')
plt.plot(rmse_x, rmse_y)
#plt.yscale('log')
for i in range(0, len(rmse_y)):
	print(str(rmse_x[i]) + "\t" + str(rmse_y[i]))

plt.show()
