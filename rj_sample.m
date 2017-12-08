
function [X] = rj_sample(pdf, size, dim, N)
X = zeros(size, dim);
M = 2.0;
num = 0;
total = 0;
center =  (N/2.0)*ones(dim, 1);
while(num < size)
    total = total + 1;
    y = randn(dim, 1) + center;
    u = rand;
    if(u < pdf(y) / (mvnpdf(y, center)* M))
        num = num + 1;
        X(num, :) = y;
    end
end

fprintf("Rejection Rate: %.3f", 1 - double(num) / total);

end

