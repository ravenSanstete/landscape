clear all;
load data_500;

%% just use X as the input on (NI)^n

%% define the relu-2 net
relu_2 = @(X, w) max([dot(X,w) , 0]);
D = @(X, w) double(dot(X,w) > 0);



%% the teacher network parameter
w_star = randn(n, 1);

x = X(randi([1, size(X, 1)]), :);


expected_term = D(x, w_star)*x*w_star;

delta_J = @(w) x*D(x, w)*(D(x,w)*x*w - expected_term);

disp(delta_J(randn(n,1)));

