clear all;
load data_5000;

%% just use X as the input on (NI)^n

%% define the relu-2 net
relu_2 = @(X, w) max([dot(X,w) , 0]);
D = @(X, w) double(dot(X,w) > 0);
J = @(X, w, w_star) (relu_2(X, w) - relu_2(X, w_star))^2;



%% the teacher network parameter
w_star = randn(n, 1);
sp_size = 5000;
param = zeros(sp_size, n);
loss = zeros(sp_size, 1);

for iter = [1:sp_size]
    
x = X(randi([1, size(X, 1)]), :);
param(iter, :) = randn(n, 1);
loss(iter) = J(x, param(iter,:), w_star);
% expected_term = D(x, w_star)*x*w_star;
% 
% delta_J = @(w) x*D(x, w)*(D(x,w)*x*w - expected_term);
end

param_Y = tsne(param);

plot3(param_Y(:, 1), param_Y(:, 2), loss, '.');
