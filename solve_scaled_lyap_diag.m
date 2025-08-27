function Psol = solve_scaled_lyap_diag(A_cl, rho)
n = size(A_cl,1);
P = sdpvar(n, n);

% LMI constraints
LMI = [P >= 1e-2*eye(n), A_cl*P*A_cl' - (1-rho)*P <= 0];

% Objective: sum of absolute off-diagonal elements
obj = sum(sum(abs(P - diag(diag(P)))));

options = sdpsettings('solver','sdpt3','verbose',1);
diagnostics = optimize(LMI, obj, options);

if diagnostics.problem == 0
    Psol = value(P);
    disp('Most diagonal P found:');
    disp(Psol)
else
    disp('No feasible solution');
end
end
