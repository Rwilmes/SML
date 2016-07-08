import numpy as np
import matplotlib.pyplot as plt




X = np.loadtxt('../dataSets/mnist_small_train_in.txt', delimiter=',').T
Y_train = np.loadtxt('../dataSets/mnist_small_train_out.txt', delimiter=',')
Y_train = np.array([Y_train])

X_test = np.loadtxt('../dataSets/mnist_small_test_in.txt', delimiter=',').T
Y_test = np.loadtxt('../dataSets/mnist_small_test_out.txt', delimiter='.')

numiter = 1000
numHidden = 10
learningrate = 0.001
bias = -1

numOutput = Y_train.shape[0]
numInput = X.shape[0]
N = X.shape[1]

add = -np.ones([1,N])
X = np.concatenate((add, X), axis=0)
Wx = np.random.rand(numHidden, numInput+1)
Wy = np.random.rand(numOutput, numHidden+1)

errors = np.zeros((numiter,1))

for i in np.arange(0,numiter):
    # Calculate output values for hidden layers. Use sigmoid function
    Z = np.dot(Wx, X)
    Z = 1+np.exp(-Z)
    Z = np.divide(1, Z)

    # Add constant
    S = np.concatenate((add, Z), 0)

    # Calculate output of output neuron. Use linear function
    Y = np.dot(Wy, S)

    # Calculate difference between expected and achieved value
    E = np.subtract(Y_train, Y)

    # Calculate rmse
    error = np.mean(np.power(E, 2))
    errors[i] = error

    # Calculate next weights for second layer
    dWy = np.dot(E, S.T)
    dWy = np.divide(dWy, N)
    dWy = np.multiply(learningrate, dWy)
    Wy = np.add(Wy, dWy)

    # Calculate next weights for input layer
    dPhi = np.multiply(S, np.subtract(1, S))
    dWx = np.dot(Wy.T, E)
    dWx = np.multiply(dPhi, dWx)
    dWx = dWx[1:,:]
    dWx = np.dot(dWx, X.T)
    dWx = np.divide(dWx, N)
    dWx = np.multiply(learningrate, dWx)
    Wx = np.add(Wx, dWx)

add = -np.ones([1,X_test.shape[1]])
X_test = np.concatenate((add, X_test), axis=0)
Z = np.dot(Wx, X_test)
Z = 1 + np.exp(-Z)
Z = np.divide(1, Z)
S = np.concatenate((add, Z), 0)
Y = np.dot(Wy, S)
print Y.shape, Y_test.shape
plt.figure()
plt.plot(Y.T, label='Y predicted')
plt.plot(Y_test.T, label='Y original')
plt.legend()

# print errors
plt.figure();
plt.plot(errors)
plt.xlabel('Iteration #')
plt.ylabel('Error')
plt.title('Plot of the mean squared error')
plt.show()