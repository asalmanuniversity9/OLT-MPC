function [Psol, diagnostics] = solve_diag_lyap_min(A_cl, rho)
% Find diagonal positive definite P minimizing trace(P)
% subject to A_cl' * P * A_cl - (1-rho) * P <= 0

n = size(A_cl,1);
P_diag = sdpvar(n,1);             % Diagonal entries of P
P = diag(P_diag);                 % P is diagonal

% LMI constraint and positivity
LMI = [P_diag >= 1e-6, ...
       A_cl*P*A_cl' - (1-rho)*P <= -1e-8,...
       %norm(P_diag,2) <= 100,...
       ];

% Objective: minimize sum(P_diag) = trace(P)
objective = sum(P_diag);

options = sdpsettings('solver','sdpt3','verbose',0);
diagnostics = optimize(LMI, objective, options);

if diagnostics.problem == 0
    Psol = value(P);
else
    Psol = [];
    warning('No feasible solution');
end
end
