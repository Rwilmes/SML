import matplotlib.pyplot as plt
import numpy as np


def exprepeat(x_in, n):
    X_out = np.ones([x_in.size, 1])

    for i in range(0, degree):
        v = np.array([X_out[:, 0]]).transpose()
        v = np.multiply(v, x_in)
        X_out = np.concatenate((v, X_out), axis=1)

    X_out = X_out.transpose()

    #print "X_out", X_out[:,0]
    #print "X_in", x_in.transpose()
    return X_out


def polyregress(x,y,degree):
    X_p = exprepeat(x, degree)

    w = np.linalg.inv(np.dot(X_p, X_p.transpose()))
    w = np.array([np.dot(np.dot(w, X_p), y)])

    return w, X_p


data = np.loadtxt("../dataSets/linRegData.txt")
test = data[:,1]
trainsize = 20

x_train = np.array([data[0:trainsize,0]]).transpose()
y_train = np.array(data[0:trainsize,1]).transpose()

data = data[trainsize:,:]
data = data[data[:,0].argsort()]
x_test = np.array([data[:,0]]).transpose()
y_test = np.array([data[:,1]]).transpose()

plt.close("all")
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')

rmse_test_t = np.array([[]])
rmse_train_t = np.array([[]])
for degree in range(0,21):
    w, X_train = polyregress(x_train, y_train, degree)

    X = exprepeat(x_test, degree)

    ye_train = np.dot(w, X_train).transpose()
    ye_test = np.dot(w, X).transpose()

    # Calculate RMSE of test data
    rmse_test = np.power(np.subtract(y_test, ye_test),2)
    rmse_test = np.array([[np.divide(np.sum(rmse_test), rmse_test.size)]])
    rmse_test_t = np.concatenate((rmse_test_t, rmse_test), axis=1)
    # and train data
    rmse_train = np.power(np.subtract(y_train, ye_train),2)
    rmse_train = np.array([[np.divide(np.sum(rmse_train), rmse_train.size)]])
    rmse_train_t = np.concatenate((rmse_train_t, rmse_train), axis=1)

    print "RMSE of testset with degree {}: {}".format(degree, rmse_test)
    print "RMSE of train with degree {}: {}".format(degree, rmse_train)

    # y2_train = np.dot(w, X_train).transpose()
    plt.plot(x_test, ye_test, label='degree {}'.format(degree))
    plt.plot(x_train, ye_train, '--')

plt.legend()

plt.figure()
xplt = np.array([np.arange(0,21)]).transpose()
plt.plot(xplt, rmse_test_t.transpose(), label='RMSE of test set')
plt.plot(xplt, rmse_train_t.transpose(), label='RMSE of train set')
plt.legend()

plt.show()

