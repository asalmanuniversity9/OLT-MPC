function [maxNorm, xStar] = maxKernelNorm_bounded(R, ell, sigma, lb, ub)
% MAXKERNELNORM  Maximise ‖k_R(x)‖₂ for a squared-exponential kernel
%                with OPTIONAL box constraints  lb ≤ x ≤ ub.
%
%   [maxNorm, xStar] = maxKernelNorm(R, ell, sigma)
%   [maxNorm, xStar] = maxKernelNorm(R, ell, sigma, lb, ub)
%
% Inputs
%   R     : d-by-m matrix, reference points
%   ell   : positive scalar, length-scale
%   sigma : positive scalar, signal std-dev  (k = σ² exp(−‖·‖²/(2ℓ²)))
%   lb    : (optional) d-vector lower bound  (use [] or omit ⇒ −Inf)
%   ub    : (optional) d-vector upper bound  (use [] or omit ⇒ +Inf)
%
% Outputs
%   maxNorm : maximum L2 norm ‖k_R(x)‖₂
%   xStar   : arg-max location (d-vector)

% ------------------------------------------------------ argument handling
d = size(R,1);
if nargin < 4 || isempty(lb),  lb = -inf(d,1);  end
if nargin < 5 || isempty(ub),  ub =  inf(d,1);  end
lb = lb(:);  ub = ub(:);
assert(all(size(lb)==[d,1]) && all(size(ub)==[d,1]), ...
       'lb and ub must be d-vectors (or []).')
assert(all(lb <= ub), 'Each lb must not exceed the corresponding ub.')

% ---------------------------------------------------------- objective
fneg = @(x) -sigma^4 * sum(exp(-sum((x(:)-R).^2,1)/ell^2));   % −f(x)

unbounded = all(isinf(lb) & lb<0) && all(isinf(ub) & ub>0);

% ------------------------------------------------------ optimisation
if unbounded
    % --- unconstrained: use fminunc (no bounds to worry about)
    opts = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
    x0   = mean(R,2);
    [xStar,fval] = fminunc(fneg,x0,opts);
else
    % --- bounded: use fmincon or fall back to projected patternsearch
    if license('test','Optimization_Toolbox')
        opts = optimoptions('fmincon','Algorithm','interior-point',...
                            'Display','off');
        x0   = max(min(mean(R,2),ub),lb);
        [xStar,fval] = fmincon(fneg,x0,[],[],[],[],lb,ub,[],opts);
    else
        % light-weight fallback: projected mean-shift (few iterations)
%         x = max(min(R(:,1),ub),lb);              % start at first ref-point
%         for t = 1:200
%             w  = exp(-sum((x-R).^2,1)/ell^2);
%             x  = (R*w')/sum(w);                  % mean-shift update
%             x  = max(min(x,ub),lb);              % project back to box
%         end
        xStar = x;   fval = fneg(xStar);
    end
end

maxNorm = sqrt(-fval);
end
