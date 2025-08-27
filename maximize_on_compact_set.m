function [x_max, f_max, exitflag, output] = maximize_on_compact_set(fun, lb, ub, x0, options)
% maximize_on_compact_set Maximizes a function over a compact (box-constrained) set.
%
%   [x_max, f_max, exitflag, output] = maximize_on_compact_set(fun, lb, ub, x0, options)
%
%   Inputs:
%     fun      - Function handle to maximize, accepts vector input
%     lb       - Lower bounds (vector)
%     ub       - Upper bounds (vector)
%     x0       - Initial guess (vector)
%     options  - (optional) Options for fmincon (optimoptions('fmincon',...))
%
%   Outputs:
%     x_max    - Maximizer of the function
%     f_max    - Maximum value of the function
%     exitflag - fmincon exit flag
%     output   - fmincon output structure
%
%   Example:
%     fun = @(x) -norm(x)^2;        % Maximize -(x1^2+x2^2) over [-1,1]^2
%     lb = [-1; -1];
%     ub = [1; 1];
%     x0 = [0.5; 0.5];
%     [x_max, f_max] = maximize_on_compact_set(fun, lb, ub, x0)

if nargin < 5 || isempty(options)
    options = optimoptions('fmincon', 'Display', 'off');
end

neg_fun = @(x) -fun(x);  % Because fmincon minimizes

[x_max, neg_f_max, exitflag, output] = fmincon(neg_fun, x0, [], [], [], [], lb, ub, [], options);

f_max = fun(x_max);      % Maximum value
end

