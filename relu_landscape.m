clear all;
load data_500;


%% define the relu-2 net
relu_2 = @(X, w) max([(X*w)';zeros(1, size(X, 1))]);