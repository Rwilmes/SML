clear variables;

X = load('mnist_small_train_in.txt');
X = [ones(size(X,1) ,1), X];
y = load('mnist_small_train_out.txt');
X_test = load('mnist_small_test_in.txt');
y_test = load('mnist_small_test_out.txt');

num_h = 5;

[Wx, Wy, MSE] = trainMLP(size(X,2), 10, 1, 1, 0.001, X', y', 1000, 0.1);

semilogy(MSE);
W1 = 0.1*ones(num_h, size(X, 1)+1);
W2 = 0.1*ones(size(y,1), num_h+1);

for i=1:10
    z = W1*X;
    yhat = sum(W2*z, 2);
    
    error2 = (y-yhat);
    error1 = W2' * error2;
    
    dW2 = error2 * z';
    dW1 = W1.*repmat(error1, 1, size(W1,2));
    
    W2 = W2 - 0.1*dW2;
    W1 = W1 - 0.1*dW1;
    
    e = sqrt(sum(error2.^2)/length(error2));
    disp(e);
end