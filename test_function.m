n = size(A_cl,1);
N_rho = 20;
rhos = linspace(0.05, 0.95, N_rho);
gammas = nan(N_rho,1);
Pn_cells = cell(N_rho,1);

for i = 1:N_rho
    rho = rhos(i);
    P_n = sdpvar(n,n);
    gamma = sdpvar(1,1);

  LMI1 = (A_cl'*P_n*A_cl - (1 - rho)*P_n) <= 0;
LMI2 = (P_n - gamma*eye(n)) <= 0;

    constraints = [LMI1,LMI2, P_n >= 1e-6*eye(n), gamma >= 1e-6];
    objective = gamma;
    ops = sdpsettings('solver','sdpt3','verbose',0);

    sol = optimize(constraints, objective, ops);

    if sol.problem == 0
        gammas(i) = value(gamma);
        Pn_cells{i} = value(P_n);
    end
end

ratios = gammas ./ rhos';

% Only consider feasible (non-NaN) points
feasible_idx = find(~isnan(ratios));
if isempty(feasible_idx)
    error('No feasible solutions found for any rho!');
end

[optval, relidx] = min(ratios(feasible_idx)); % relidx: index in feasible_idx
best_idx = feasible_idx(relidx); % index in original rhos/gammas/Pn_cells

rho_best = rhos(best_idx);
gamma_best = gammas(best_idx);
P_n_best = Pn_cells{best_idx};

fprintf('Best gamma/rho = %g\n', optval);
fprintf('At rho = %g, gamma = %g\n', rho_best, gamma_best);
disp('Optimal P_n:');
disp(P_n_best);
