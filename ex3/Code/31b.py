import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

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

var = 0.02

space = np.linspace(0, 2, num=20)
x = np.linspace(0, 2, num=201)

matrix = get_phi_gaussians(space, x, var)
matrix = normalize(matrix)
plot_phi(matrix, x)
