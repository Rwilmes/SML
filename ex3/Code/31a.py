import matplotlib.pyplot as plt
import numpy as np


def exprepeat(x_in, n):
    X_out = np.ones([x_in.size, 1])

    for i in range(0, degree):
        v = np.array([X_out[:, 0]]).T
        v = np.multiply(v, x_in)
        X_out = np.concatenate((v, X_out), axis=1)

    X_out = X_out.T
    return X_out


def polyregress(x, y, degree):
    X_p = exprepeat(x, degree)

    w = np.linalg.solve(np.dot(X_p, X_p.transpose()), X_p)
    w = np.array([np.dot(w, y)])

    return w, X_p


def rmse(y, y_fit):
    e = np.subtract(y, y_fit) ** 2
    e = np.sqrt([[np.divide(np.sum(e), e.size)]])
    return e


data = np.loadtxt("../dataSets/linRegData.txt")
test = data[:, 1]
trainsize = 20

x_all = np.array([np.arange(0, 2, 0.001)]).T
x_all = np.sort(x_all, axis=0)

x_train = np.array([data[0:trainsize, 0]]).T
y_train = np.array(data[0:trainsize, 1]).T

data = data[trainsize:, :]
data = data[data[:, 0].argsort()]
x_test = np.array([data[:, 0]]).T
y_test = np.array([data[:, 1]]).T

plt.close("all")
plt.plot(x_train, y_train, 'o', label="Train set")
plt.plot(x_test, y_test, 'x', label="Test set")

rmse_test_t = np.array([[]])
rmse_train_t = np.array([[]])
for degree in range(17, 18):
    w, X_train = polyregress(x_train, y_train, degree)

    X = exprepeat(x_test, degree)
    X_train = exprepeat(x_train, degree)
    X_all = exprepeat(x_all, degree)

    ye_all = np.dot(w, X_all).T
    ye_train = np.dot(w, X_train).T
    ye_test = np.dot(w, X).T

    # Calculate RMSE of test data
    # rmse_test = np.subtract(y_test, ye_test) ** 2
    # rmse_test = np.sqrt([[np.divide(np.sum(rmse_test), rmse_test.size)]])
    rmse_test = rmse(y_test, ye_test)
    rmse_test_t = np.concatenate((rmse_test_t, rmse_test), axis=1)
    # and train data
    # rmse_train = np.subtract(np.array([y_train]).T, ye_train) ** 2
    # rmse_train = np.sqrt([[np.divide(np.sum(rmse_train), rmse_train.size)]])
    rmse_train = rmse(np.array([y_train]).T, ye_train)
    rmse_train_t = np.concatenate((rmse_train_t, rmse_train), axis=1)

    # print "RMSE of testset with degree {}: {}".format(degree, rmse_test)
    # print "RMSE of train with degree {}: {}".format(degree, rmse_train)

    # y2_train = np.dot(w, X_train).T
    plt.plot(x_all, ye_all, label='degree {}'.format(degree))
    #plt.plot(x_train, ye_train, '--')

plt.grid()
plt.legend(loc='best')

f = plt.figure()
xplt = np.array([np.arange(17, 18)]).T
plt.plot(xplt, rmse_test_t.T, label='RMSE of test set')
plt.plot(xplt, rmse_train_t.T, label='RMSE of train set')
plt.grid()
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE")
f.gca().xaxis.set_ticks(np.arange(start=0, stop=22, step=1))
f.gca().yaxis.set_ticks(np.arange(start=0, stop=1.2, step=0.05))
plt.legend(loc='best')

plt.show()
