%% consider a distribution on (NI)^n with translation distribution I = [0,1]
MORINO = [false, true];
n = 3;
center = 0.5 * ones(n, 1);
s = 0.0;
e = 1.0;
N = 3;
sample_size = 500;
sigma = 0.05;

g = makedist('Normal', 0 , sigma);
normalizer = 1.0/ ((cdf(g, 0.5) - cdf(g, -0.5)) ^ n) ;

%% the truncated gaussian for distribution over I^3
p_i = @(x) normalizer * mvnpdf(x, center, sigma^2*eye(n)).* all(and(x < e, x >= s));

normalizer_2 = 1.0 / (n^N);
%% say x from (NI)^n
p = @(x) normalizer_2 * p_i(x - floor(x)) .* all(and(x < N, x >= 0));

%% TEST the shape of the translation invariant distribution
if(MORINO(1))

    X = N * rand(n, sample_size);
    density = [];

    iter = 1;

    for pt = X
        density(iter) = p(pt);
        iter = iter + 1;
    end

    Y = tsne(X');

    plot3(Y(:, 1), Y(:, 2), density, '.');
end

%% Next, we do some sampling from the given distribution
X = rj_sample(p, sample_size, n, N);

%% Test the shape of the sampling
if(MORINO(2))
   plot3(X(:, 1), X(:, 2), X(:, 3), '.');   
end


