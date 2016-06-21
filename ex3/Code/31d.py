import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
from numpy.linalg import inv

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

data = np.loadtxt("../dataSets/linRegData.txt")
var = 0.0025

NS=[10, 12, 16, 20, 50, 150]
NS=[20]

degree = 12

for N in NS:
	x_test = data[N:,0]
	x_train = data[0:N,0]
	y_test = data[N:,1]
	y_train = data[0:N,1]
	x_train_v = np.array([x_train]).transpose()
	x_test_v = np.array([x_test]).transpose()
	y_train_v = np.array([y_train]).transpose()
	y_test_v = np.array([y_test]).transpose()


	phi = get_phi_polynomial(x_train, degree).T
	X = get_phi_polynomial(x_test, degree)

	# w = pseudoinverse of phi
	w = phi.T.dot(phi)
	w = np.linalg.inv(w)
	w = w.dot(phi.T).T

	# calc test preditctions
	ye_test = w.dot(X).T

	# calc deviation at each x
	deviation = []
	for i in range(0, len(x_test)):
		x = x_test[i]
		dev = 0
		for j in range(0, len(ye_test[i])):
			dev += (x-ye_test[i][j])**2
		dev = dev/len(ye_test[i])
		dev = math.sqrt(dev)
		deviation.append(dev)

	# calc upper and lower bound of deviation and mean at each x
	y_mean = []
	dev_upper = []
	dev_lower = []
	for i in range(0, len(x_test)):
		y = y_test[i]
		n = len(y_test)
		devi = 1.96*deviation[i]/n
		y_mean.append(y + np.sum(ye_test[i])/len(ye_test))
		dev_upper.append(y+devi)
		dev_lower.append(y-devi)

	###
	## PLOTTING
	#
	plt.figure(N)
	plt.title("N = "+str(N) + " samples")
	plt.xlabel('x')
	plt.ylabel('y')	

	# plot data points
	plt.plot(x_train, y_train, 'o')
	#plt.plot(x_test, y_test, 'x')

	# sort points
	x_s, y_dev_lower_s = get_plot_points(x_test, dev_lower)
	x_s, y_dev_upper_s = get_plot_points(x_test, dev_upper)
	x_s, y_mean_s = get_plot_points(x_test, y_mean)
	
	# plot mean
	plt.plot(x_s, y_mean_s, color='blue')

	# plot confidence interval
	plt.fill_between(x_s, y_dev_lower_s, y_dev_upper_s, facecolor='blue', alpha=0.3)
	
plt.show()

