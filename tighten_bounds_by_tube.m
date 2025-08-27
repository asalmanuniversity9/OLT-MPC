function [lb_tight, ub_tight, R,delta,nu] = tighten_bounds_by_tube(lb, ub, C1, mu, Gamma, Sigma, C2, n_D, P, B_norm,gamma_l,inv_rho)
% Tighten box constraints lb, ub on x by tube around the error e.
% Tube:   ||e||_P <= C1*(||mu|| + gamma*sqrt(norm(Sigma))) + C2*sqrt(n_D)
%
% Inputs:
%   lb, ub   - vectors (nx1): lower and upper bounds on x
%   C1, gamma, C2, n_D, P - tube parameters as defined
%   mu       - mean error (nx1)
%   Sigma    - variance (scalar or nxn)
%
% Outputs:
%   lb_tight, ub_tight - new tightened bounds (nx1)
%   R - tube radius

    % --- 1. Compute tube radius
    if ismatrix(Sigma) && size(Sigma,1)>1
        sigma_norm = norm(Sigma,2);   % spectral norm (largest eigenvalue)
    else
        sigma_norm = Sigma;           % if scalar
    end
    %R = C1 * (norm(mu) + Gamma * sqrt(sigma_norm)) + C2 * sqrt(n_D);
    nu = sqrt(Gamma+(n_D)*0.81)*sqrt(C1*sqrt(sigma_norm)+25e-8);
    R =C1*norm(mu)+ sqrt(Gamma+(n_D)*0.81)*sqrt(C1*sqrt(sigma_norm)+25e-8);
    % --- 2. Componentwise margins (delta)
    P_inv = inv(P);
    delta = gamma_l*B_norm*sqrt(R)* sqrt(diag(P_inv))*inv_rho;

    % --- 3. Tighten constraints
    lb_tight = lb + delta';
    ub_tight = ub - delta';

    % --- 4. Check feasibility
    if any(lb_tight > ub_tight)
        error('Tube too large: tightened bounds are infeasible.');
    end
end
