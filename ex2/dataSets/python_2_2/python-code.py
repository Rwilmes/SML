import sys
import numpy as np
import math

def calc_mu(vectors):
	mu = np.zeros((2,1))
	N = len(vectors)
	for v in vectors:
		splitted = v.split()
		if(len(splitted) > 0): # skip empty lines
			mu[0] += float(splitted[0])
			mu[1] += float(splitted[1])			
	return mu/N


def calc_coVar(vectors, mu, unbiased=1):
	var = []
	N = len(vectors)

	coVarM = np.zeros((2,2))

	for v in vectors:
		splitted = v.split()
		if(len(splitted) > 0): # skip empty lines
			diff_x = float(splitted[0]) - mu[0]
			diff_y = float(splitted[1]) - mu[1]

			coVarM[0][0] += (diff_x * diff_x)
			coVarM[0][1] += (diff_x * diff_y)
			coVarM[1][0] += (diff_y * diff_x)
			coVarM[1][1] += (diff_y * diff_y)
	return coVarM/(N-unbiased)

input_path = sys.argv[1]
readFile = open(input_path, 'r')
vectors = readFile.read().split("\n")

mu = calc_mu(vectors)
coVar = calc_coVar(vectors, mu, unbiased=0)
coVarUnbiased = calc_coVar(vectors, mu, unbiased=1)

print("INPUT DATA: " + str(input_path))
print("mean:")
print(mu)
print("unbiased covar:")
print(coVar)
print("biased covar:")
print(coVarUnbiased)