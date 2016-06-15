import matplotlib.pyplot as plt
import numpy as np

def get_plot_points(x, y):
	points = {}
	for i in range(0, len(x_test)):
		points[x_test[i]]=ye_test[i][0]
	x_sorted = []
	y_sorted = []
	for x in sorted(points.keys()):
		x_sorted.append(x)
		y_sorted.append(points[x])
	return x_sorted, y_sorted

def get_phi_polynomial(x, degree):
	phi_temp = np.zeros((degree, len(x)))

	for i in range(0, degree):
		for j in range(0, len(x)):
			l =  np.power(x[j], i)
			phi_temp[degree-(i+1)][j] = l
	return phi_temp


def get_w(y, phi):
	w = np.linalg.inv(np.dot(phi, phi.transpose()))
	w = np.array([np.dot(np.dot(w, phi), y)])
	return w

def calc_rmse(thruth, estimate):
	rmse_test = np.power(np.subtract(thruth, estimate),2)
	rmse_test = np.divide(np.sum(rmse_test), rmse_test.size)
	return rmse_test

# load data
data = np.loadtxt("../dataSets/linRegData.txt")
trainsize = 20
dimension = 21

x_test = data[trainsize:,0]
x_train = data[0:trainsize,0]

y_test = data[trainsize:,1]
y_train = data[0:trainsize,1]

# plot data points
#plt.plot(x_train, y_train, 'o')
#plt.plot(x_test, y_test, 'x')

rmse_x = []
rmse_y = []

# calc curves and plot them
for degree in range(0, dimension):
	phi = get_phi_polynomial(x_train, degree)
	w = get_w(y_train, phi)
	X = get_phi_polynomial(x_test, degree)

	# fit curve
	ye_test = np.dot(w, X).transpose()

	# sort points for linespoint plot
	x_sorted, y_sorted = get_plot_points(x_test, ye_test)

	# plot
	#plt.plot(x_sorted, y_sorted)

	rmse = calc_rmse(y_test, ye_test)
	rmse_x.append(degree)
	rmse_y.append(rmse)
	print("degree: " + str(degree) + "  RMSE: " + str(rmse)) 

plt.plot(rmse_x, rmse_y)

plt.show()

