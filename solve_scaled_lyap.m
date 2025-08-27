function Psol = solve_scaled_lyap(A_cl, rho)
%SOLVE_SCALED_LYAP  Finds P > 0 such that:
%   A_cl' * P * A_cl - (1 - rho) * P <= 0
% Inputs:
%   A_cl : n x n matrix
%   rho  : scalar, 0 < rho < 1
% Output:
%   Psol : solution matrix (if feasible); [] if infeasible

    % Check inputs
    if nargin < 2
        error('Not enough inputs');
    end
    if ~(isscalar(rho) && rho > 0 && rho < 1)
        error('rho must be scalar with 0 < rho < 1');
    end
    n = size(A_cl,1);
    if size(A_cl,1) ~= size(A_cl,2)
        error('A_cl must be square');
    end

    % Setup LMI (requires YALMIP)
    P = sdpvar(n,n);
    LMI = [P >= 1e-6*eye(n), A_cl*P*A_cl' - (1 - rho)*P <= 0];
    options = sdpsettings('solver','sdpt3','verbose',0);

    diagnostics = optimize(LMI,[],options);

    if diagnostics.problem == 0
        Psol = value(P);
    else
        Psol = [];
        warning('No feasible solution found');
    end
end
