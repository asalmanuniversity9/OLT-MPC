function [theta_s, theta_no, u_s, u_no] = Equilibrium_estimator( ...
    A, B, K, X_ref, lb_tight, ub_tight, initdata, iK_RR, ytrain, opts)
% EQUILIBRIUM_ESTIMATOR (thin wrapper)
% Uses your own:
%   - find_equilibrium_ss(A,B, -X_ref, lb, ub, opts_fe)
%   - solve_equilibrium_gp(Acl, B, theta_no, ytrain, initdata, iK_RR, optsgp)
%
% Inputs:
%   A,B,K,X_ref, lb_tight, ub_tight, initdata, iK_RR, ytrain (as in your code)
%   opts fields used:
%     opts.u_lb, opts.u_ub, opts.lambda_u, opts.W
%     opts.idx_g, opts.solver, opts.Display
%
% Outputs:
%   theta_s, theta_no, u_s, u_no

% ---- light checks / shaping ----
n = size(A,1);
if size(X_ref,2) > 1, X_ref = X_ref(:); end
if size(lb_tight,2)>1, lb_tight = lb_tight(:); end
if size(ub_tight,2)>1, ub_tight = ub_tight(:); end

% ---- Step 1: nominal equilibrium (theta_no, u_no) ----
opts_fe = struct( ...
    'u_lb',     opts.u_lb, ...
    'u_ub',     opts.u_ub, ...
    'lambda_u', opts.lambda_u, ...
    'W',        opts.W);

% Note: your find_equilibrium_ss expects -X_ref (as in your snippet)
[theta_no, u_no, ~] = find_equilibrium_ss( ...
    A, B, X_ref, lb_tight, ub_tight, opts_fe);


% ---- Step 2: GP-corrected equilibrium theta_s ----
Acl = A - B*K;



optsgp = struct;
optsgp.idx_g   = opts.idx_g(:);  % e.g., [3;4]
optsgp.x0      = theta_no;       % start at nominal
optsgp.solver  = opts.solver;    % 'fsolve' or 'lsqnonlin'
optsgp.Display = opts.Display;   % 'off' etc.

[theta_s, ~] = solve_equilibrium_gp( ...
    Acl, B, theta_no, ytrain, initdata, iK_RR, optsgp);

% ---- Step 3: corrected input ----
u_s = K*(theta_s - theta_no) + u_no;

% (optional) sanity: size consistency
assert(all(size(theta_s)==[n,1]) && all(size(theta_no)==[n,1]), ...
    'theta_s/theta_no must be %d-by-1.', n);
end
