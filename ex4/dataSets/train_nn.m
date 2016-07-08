clear variables;

X = load('mnist_small_train_in.txt')';

y_D = load('mnist_small_train_out.txt')';
X_test = load('mnist_small_test_in.txt');
y_test = load('mnist_small_test_out.txt');

num_h = 100;
N = size(X,2);
mu = 0.001;
numiter = 5000;
p = size(X,1);
numOutput=1;


[Wx_other, Wy_other, MSE] = trainMLP(p, 10, numOutput, mu, 0, X, y_D, numiter, 0.1);

%semilogy(MSE);
X = [-1*ones(1,N); X];
Wx = rand(num_h, p+1);
Wy = rand(numOutput, num_h+1);

for i=1:numiter
    

    V = Wx*X;
    Z = 1./(1+exp(-V));
    
    S = [-1*ones(1,N); Z];
    Y = Wy*S;
    %Y = 1./(1+exp(-Y));
    
    E = (y_D-Y);
    
    e = mean(E.^2);
    disp(['iter: ' num2str(i) ' mse= ' num2str(e) '-' num2str(mean(e))]);
    
    %dPhi = Y.*(1-Y);
    dPhi = ones(size(Y));
    dGy = dPhi.*E;
    dWy = dGy * S';
    Wy = Wy + (mu/N)*dWy;
    
    
    dPhi = S .*(1-S);
    dGx = dPhi.*(Wy' * dGy);
    dGx = dGx(2:end,:);
    dWx = dGx*X';
    %dW1 = dW1(2:end, :); % Remove weights for z_0
    
    Wx = Wx + (mu/N)*dWx;
    
    
end