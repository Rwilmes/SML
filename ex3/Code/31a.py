import matplotlib.pyplot as plt
import numpy as np


def exprepeat(x_in, n):
    X_out = np.ones([x_in.size, 1])

    for i in range(0, degree):
        v = np.array([X_out[:, 0]]).transpose()
        v = np.multiply(v, x_in)
        X_out = np.concatenate((v, X_out), axis=1)

    X_out = X_out.transpose()

    print "X_out", X_out[:,0]
    print "X_in", x_in.transpose()
    return X_out


def polyregress(x,y,degree):
    X_p = exprepeat(x, degree)

    w = np.linalg.inv(np.dot(X_p, X_p.transpose()))
    w = np.array([np.dot(np.dot(w, X_p), y)])

    return w, X_p


data = np.loadtxt("../dataSets/linRegData.txt")
trainsize = 20

x_train = np.array([data[0:trainsize,0]]).transpose()
y_train = np.array(data[0:trainsize,1]).transpose()

data = data[trainsize:,:]
data = data[data[:,0].argsort()]
x_test = np.array([data[:,0]]).transpose()
y_test = np.array([data[:,1]]).transpose()

plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')

for degree in range(21,22):
    w, X_train = polyregress(x_train, y_train, degree)
    print "w: ", w
    #X = np.ones([data.shape[0], 1])
    #for i in range(0, degree):
    #    v = np.array([X[:, i]]).transpose()
    #    v = np.multiply(v, x_test)
    #    X = np.concatenate((X, v), axis=1)
    #X = X.transpose()

    X = exprepeat(x_test, degree)


    y2 = np.dot(w, X).transpose()
    y2_train = np.dot(w, X_train).transpose()
    plt.plot(x_test, y2, label='degree {}'.format(degree))
    plt.plot(x_train, y2_train, '--')


plt.legend()
plt.show()

