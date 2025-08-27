function [maxNorm, xStar] = maxKernelNorm(R, ell, sigma)
% R : d-by-m matrix of reference points
% ell, sigma : kernel parameters

d      = size(R,1);
fneg   = @(x) -sigma^4 * sum(exp(-sum((x(:)-R).^2,1)/ell^2));  % −f(x)
opts   = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');

% try mean of points first; restart from each x_j if you want to be thorough
[xStar, fval] = fminunc(fneg, mean(R,2), opts);

maxNorm = sqrt(-fval);   % = ‖k_R(xStar)‖₂
end
