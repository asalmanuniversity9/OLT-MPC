function [rkhs_ub, Bdelta] = gpr_rkhs_upper_bound(gprMdl, delta)
%GPR_RKHS_UPPER_BOUND  Upper bound(s) on the RKHS norm of GP posterior mean.
%   rkhs_ub = GPR_RKHS_UPPER_BOUND(gprMdl)
%      returns the standard upper bound on the RKHS norm of the learned function.
%
%   [rkhs_ub, Bdelta] = GPR_RKHS_UPPER_BOUND(gprMdl, delta)
%      also returns a high-probability bound Bdelta such that
%      ||g||_H ≤ Bdelta with probability at least 1-delta
%      (see Srinivas et al., GP-UCB, JMLR 2012).
%
%   Inputs:
%     gprMdl   - Trained RegressionGP model (from fitrgp)
%     delta    - (optional) probability error, e.g. 0.05 for 95% confidence
%
%   Outputs:
%     rkhs_ub  - data-dependent upper bound
%     Bdelta   - high-probability upper bound (if delta supplied)

    % Data-dependent RKHS upper bound
    rkhs_ub = sqrt(gprMdl.Y' * gprMdl.Alpha);

    % High-probability upper bound (if requested)
    if nargin < 2 || isempty(delta)
        Bdelta = [];
    else
        sigma_n = gprMdl.Sigma;  % estimated noise std (scalar)
        Bdelta = rkhs_ub + sigma_n * sqrt(2 * log(1/delta));
    end
end
