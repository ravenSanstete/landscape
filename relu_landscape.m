clear all;
load data_5000;

%% just use X as the input on (NI)^n
X = X - N/2.0;

%% define the relu-2 net
relu_2 = @(X, w) max([dot(X,w) , 0]);
D = @(X, w) double(dot(X,w) > 0);
J = @(X, w, w_star) (relu_2(X, w) - relu_2(X, w_star))^2;


%% now X \in \mathbb{R}^{N*d}
relu_2_batch = @(X, w) max([X*w ; zeros(N, 1)]);
D_batch = @(X, w) double(diag(relu_2_batch(X, w))); % return a matrix of N * N, diag
J_batch = @(X, w, w_star) mean((relu_2_batch(X, w) - relu_2_batch(X, w_star))^2);


delta_J_batch = @(X, w, w_star) X'*D_batch(X, w)*(D_batch(X,w)*X*w -  D_batch(X, w_star)*X*w_star);


count = 0;
prober_limit = 100;


batch_size = 500;
bound = 0.001;

TRANSLATION = false;
ROTATION = true;
BROWNIAN = false;

%% the teacher network parameter
w_star = rand(n, 1);
p_sigma = 3;

norms = [];
losses = [];
iter = 0;
BOUND = 1000;


if(TRANSLATION)
direction = rand(n, 1);
direction = direction ./ norm(direction);
% direction = direction - dot(direction, w_star)*w_star; % do some projection
interval = [-BOUND:0.05:BOUND];

for pt = interval
iter = iter + 1;

w = w_star + pt * direction;


prober = X(randsample(sample_size, batch_size),:);
grad = delta_J_batch(prober, w, w_star)/batch_size;
loss = J_batch(prober, w, w_star);
losses(iter) = loss;
norms(iter) = norm(grad);
end

plot(interval, norms);
figure;
plot(interval, losses);
end



if(ROTATION)
interval = [1:BOUND];
radians = 2*pi/BOUND;
rot_mat = makehgtform('xrotate', radians);
rot_mat = rot_mat(1:3, 1:3);

% rot_mat = RandOrthMat(n, 1e-6);
pts = zeros(size(interval, 1), 3);
w = w_star;
for pt = interval
    iter = iter + 1;
    pts(iter, :) = w;
    prober = X(randsample(sample_size, batch_size),:);
    grad = delta_J_batch(prober, w, w_star)/batch_size;
    norms(iter) = norm(grad);
    loss = J_batch(prober, w, w_star);
    losses(iter) = loss;
    w = rot_mat*w;
end

plot(interval, norms);
figure;
plot3(pts(:, 1), pts(:, 2), pts(:, 3), '.');
figure;
plot(interval, losses);
end


if(BROWNIAN)
interval = [1:BOUND];
b_sigma = 0.1;

% rot_mat = RandOrthMat(n, 1e-6);
pts = zeros(size(interval, 1), 3);
w = w_star;
for pt = interval
    iter = iter + 1;
    pts(iter, :) = w;
    prober = X(randsample(sample_size, batch_size),:);
    grad = delta_J_batch(prober, w, w_star)/batch_size;
    norms(iter) = norm(grad);
    loss = J_batch(prober, w, w_star);
    losses(iter) = loss;
    w = w + b_sigma*randn(n, 1);
end

plot(interval, norms);
figure;
plot3(pts(:, 1), pts(:, 2), pts(:, 3));
figure;
plot(interval, losses);
end