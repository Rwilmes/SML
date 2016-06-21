import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Calculate direction of separating axis using fisher discriminant analysis
def fisherw(C1, C2):
    Sw = (np.cov(C1) + np.cov(C2)).T
    m_d = np.array([np.mean(C1, 1) - np.mean(C2, 1)]).T
    w = np.linalg.solve(Sw, m_d)
    w = np.divide(w, np.linalg.norm(w)).T
    return w


def getStats(C):
    m = np.mean(C, axis=1)
    v = np.var(C, axis=1)
    return np.array([m, v]).T


def getProbability(x, stat):
    m = stat[0]
    v = stat[1]
    p = x - m
    p **= 2
    p = -p
    p /= (2 * v)
    # p = -((x-m)**2)/(2*v)
    p = np.exp(p)
    p *= np.divide(1, np.sqrt(2 * np.pi * v))
    return p


data = np.loadtxt('../dataSets/ldaData.txt')

# Put data into separate classes
C1 = data[0:50, :].T
C2 = data[50:100, :].T
C3 = data[100:150, :].T

# Calculate directions of separating axises
w1 = fisherw(C1, C2)[0]
w2 = fisherw(C1, C3)[0]
w3 = fisherw(C2, C3)[0]
w = np.array([w1, w2, w3])

# Project data onto axises
C1_p = np.dot(w, C1)
C2_p = np.dot(w, C2)
C3_p = np.dot(w, C3)

# Calcualte mean and variance of projected data
C1_s = getStats(C1_p)
C2_s = getStats(C2_p)
C3_s = getStats(C3_p)

data_p = np.dot(w, data.T)

C_classified = np.zeros(data.shape[0])
for i in np.arange(0, C_classified.size):
    p_12 = getProbability(data_p[0, i], C1_s[0])
    p_13 = getProbability(data_p[1, i], C1_s[1])
    p1 = p_12 * p_13

    p_21 = getProbability(data_p[0, i], C2_s[0])
    p_23 = getProbability(data_p[2, i], C2_s[2])
    p2 = p_21 * p_23

    p_31 = getProbability(data_p[1, i], C3_s[1])
    p_32 = getProbability(data_p[2, i], C3_s[2])
    p3 = p_31 * p_32

    p = np.array([p1, p2, p3])
    print p
    I = np.argmax(p)
    C_classified[i] = I

# Plot samples according to their classification
for i in np.arange(0, C_classified.size):
    if C_classified[i] == 0:
        plt.plot(data[i, 0], data[i, 1], 'ro', label='Class 1')
    if C_classified[i] == 1:
        plt.plot(data[i, 0], data[i, 1], 'go', label='Class 2')
    if C_classified[i] == 2:
        plt.plot(data[i, 0], data[i, 1], 'bo', label='Class 3')
plt.ylim([1, 5])
plt.xlim([4, 8])
plt.title('Classified data')
#plt.legend(loc='best')

# Plot samples according to their original class
w0 = [6.5, 3]
plt.figure()
plt.plot(C1[0, :], C1[1, :], 'ro', label='Class 1')
plt.plot(C2[0, :], C2[1, :], 'go', label='Class 2')
plt.plot(C3[0, :], C3[1, :], 'bo', label='Class 3')
plt.plot([w0[0], w[0, 0] + w0[0]], [w0[1], w[0, 1] + w0[1]], '-r', label='1-2')
plt.plot([w0[0], w[1, 0] + w0[0]], [w0[1], w[1, 1] + w0[1]], '-g', label='1-3')
plt.plot([w0[0], w[2, 0] + w0[0]], [w0[1], w[2, 1] + w0[1]], '-b', label='2-3')
plt.ylim([1, 5])
plt.xlim([4, 8])
plt.title('Original classes')
plt.legend(loc='best')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.plot(C1_12.T, C1_13.T)
Axes3D.scatter(ax, C1_p[0, :], C1_p[1, :], zs=C1_p[2, :], c='r', label='Class 1')
Axes3D.scatter(ax, C2_p[0, :], C2_p[1, :], zs=C2_p[2, :], c='g', label='Class 2')
Axes3D.scatter(ax, C3_p[0, :], C3_p[1, :], zs=C3_p[2, :], c='b', label='Class 3')
plt.legend(loc='best')
ax.set_xlabel('Separating 1-2')
ax.set_ylabel('Separating 1-3')
ax.set_zlabel('Separating 2-3')

plt.show()
