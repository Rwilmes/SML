clear variables;

X = load('mnist_small_train_in.txt');
y = load('mnist_small_train_out.txt');

num_h = 200;

W1 = ones(num_h, size(X, 1));
W2 = ones(size(y,1), num_h);

z = sum(W1*X, 2);
yhat = sum(W2*z, 2);

error2 = (y-yhat);
error1 = error2' * W2;