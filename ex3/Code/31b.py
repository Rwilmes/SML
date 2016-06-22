import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

def plot_phi(phi, x):
	for i in range(0, len(phi)):
		plt.plot(x, phi[i])
	plt.show()

# normalizes columns
def normalize(matrix):
	for i in range(0, len(matrix[0])):
		matrix[:,i] = matrix[:,i]/np.sum(matrix[:,i])
	return matrix

# creates phi matrix with gaussians
def get_phi_gaussians(space, x, var):
	matrix = np.zeros((len(x), len(space)))
	for i in range(0, len(matrix[0])):
		matrix[:,i] = x
	for i in range(0, len(matrix)):
		matrix[i] = matrix[i] - space
	matrix = np.exp(-np.power(matrix, 2)/(2*var))
	return matrix.T

var = 0.02

space = np.linspace(0, 2, num=20)
x = np.linspace(0, 2, num=200)

phi = get_phi_gaussians(space, x, var)
phi = normalize(phi)
plot_phi(phi, x)
