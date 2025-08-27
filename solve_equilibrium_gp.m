function [theta_s, info] = solve_equilibrium_gp(A,B, theta, mu_t, initdata, iK_RR, opts)
% SOLVE_EQUILIBRIUM_GP  Solve (I-A)(theta_s - theta) = H(theta_s)*mu_t
%
% Inputs:
%   A        : n×n system matrix
%   theta    : n×1 nominal equilibrium point (given)
%   mu_t     : M×p matrix; column j is the mean vector (over inducing points) for GP output j
%   initdata : struct with cell array gprMdl{j}, each a RegressionGP (or similar)
%              using fields: .X (inducing/representer points, size M×n_x) and .KernelInformation
%   iK_RR    : cell array with iK_RR{j} (M×M), the inverse (or solve operator) for (K_RR + noise)
%   opts     : (optional) struct
%              .idx_g      -> indices in state (1..n) where GP outputs map (length p). Default: 1:p
%              .x0         -> initial guess for theta_s. Default: theta.
%              .lb, .ub    -> bounds for theta_s (n×1). If provided, uses lsqnonlin.
%              .solver     -> 'fsolve' (default) or 'lsqnonlin'
%              .Display    -> 'off' (default), 'iter'
%
% Output:
%   theta_s  : n×1 solution
%   info     : struct with solver diagnostics

    n = size(A,1);
    if nargin < 6, opts = struct(); end
    if ~isfield(opts,'Display'),   opts.Display = 'off'; end
    if ~isfield(opts,'solver'),    opts.solver = 'fsolve'; end
    if ~isfield(opts,'x0'),        opts.x0 = theta; end

    % Number of GP outputs p inferred from cells / mu_t columns
    p = numel(initdata.gprMdl);
    assert(size(mu_t,2) == p, 'mu_t must have p columns (one per GP output).');

    % Where do GP outputs enter the state equation? default map 1..p
    if ~isfield(opts,'idx_g'), opts.idx_g = (1:p).'; end
    idx_g = opts.idx_g(:);
    assert(all(1 <= idx_g & idx_g <= n) && numel(idx_g)==p, 'idx_g must be length p with values in 1..n');

    I = eye(n);

    % Residual function F(theta_s)
    function r = F_ths(ths)
        % build H(ths)*mu_t as a p×1 vector using your formula style
        hmu = zeros(p,1);
        for j = 1:p
            mdl  = initdata.gprMdl{j};
            % mykernel_single expects (x' , X, Kinfo); ensure row vs col consistency
            kvec = mykernel_single(ths(:)', mdl.X, mdl.KernelInformation);  % 1×M
            hmu(j,1) = kvec * iK_RR{j} * mu_t(:,j);                         % scalar
        end

        rhs = zeros(n,1);
        %rhs(idx_g) = hmu;                       % embed GP contribution into selected state rows
        r = (I - A) * (ths - theta) - B*hmu;      % n×1 residual
    end

    info = struct('exitflag',NaN,'solver','','message','');

    % Choose solver (bounded or unbounded)
    use_bounds = isfield(opts,'lb') && isfield(opts,'ub') && ~isempty(opts.lb) && ~isempty(opts.ub);
    if strcmpi(opts.solver,'lsqnonlin') || use_bounds
        if ~isfield(opts,'lb'), opts.lb = -inf(n,1); end
        if ~isfield(opts,'ub'), opts.ub =  inf(n,1); end
        lsq_opts = optimoptions('lsqnonlin','Display',opts.Display,'SpecifyObjectiveGradient',false);
        [theta_s,~,resnorm,exitflag,output] = lsqnonlin(@F_ths, opts.x0, opts.lb, opts.ub, lsq_opts);
        info.exitflag = exitflag; info.solver = 'lsqnonlin';
        info.message  = output.message; info.resnorm = resnorm;
    else
        fs_opts = optimoptions('fsolve','Display',opts.Display,'SpecifyObjectiveGradient',false,...
                               'FunctionTolerance',1e-10,'StepTolerance',1e-10);
        [theta_s,~,exitflag,output] = fsolve(@F_ths, opts.x0, fs_opts);
        info.exitflag = exitflag; info.solver = 'fsolve'; info.message = output.message;
    end
end
