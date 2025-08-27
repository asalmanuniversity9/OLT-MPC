function [theta, u, info] = find_equilibrium_ss(A, B, X_ref, lb, ub, opts)
% FIND_EQUILIBRIUM_SS  Steady-state (theta,u) near X_ref with bounds on theta (and optional bounds on u)
%
%   [theta, u, info] = find_equilibrium_ss(A, B, X_ref, lb, ub, opts)
%
% Inputs:
%   A,B     : system matrices (x_{k+1} = A x_k + B u_k), A is n×n, B is n×m
%   X_ref   : desired state reference (n×1)
%   lb, ub  : elementwise lower/upper bounds on steady state theta (n×1). Use -Inf/Inf where free.
%   opts    : (optional struct)
%             .u_lb, .u_ub  -> bounds on u (m×1), default -Inf/Inf
%             .W            -> positive semidefinite weight on (theta - X_ref), default eye(n)
%             .lambda_u     -> nonnegative regularization on u, default 1e-6
%             .solver       -> 'quadprog' (default) or 'fmincon'
%
% Outputs:
%   theta,u : steady-state state and input
%   info    : struct with fields (exitflag, solver, message)
%
% Notes:
%   - If no bounds are active and B has full column rank for the target,
%     the closed-form is u = pinv(B)*((I-A)*theta), theta ≈ X_ref.
%   - Feasibility requires (I-A)*theta ∈ range(B). The optimizer finds the
%     closest feasible theta within [lb,ub].

    % ---------- sizes & defaults ----------
    n = size(A,1);
    m = size(B,2);

    if nargin < 6, opts = struct(); end
    if ~isfield(opts,'u_lb'), opts.u_lb = -inf(m,1); end
    if ~isfield(opts,'u_ub'), opts.u_ub =  inf(m,1); end
    if ~isfield(opts,'W'),     opts.W   = eye(n);    end
    if ~isfield(opts,'lambda_u'), opts.lambda_u = 1e-6; end
    if ~isfield(opts,'solver'),    opts.solver    = 'quadprog'; end

    % basic checks
    assert(all(size(A)==[n,n]), 'A must be square n×n');
    assert(all(size(B)==[n,m]), 'B must be n×m');
    assert(all(size(X_ref)==[n,1]), 'X_ref must be n×1');
    assert(all(size(lb)==[n,1]) && all(size(ub)==[n,1]), 'lb/ub must be n×1');
    assert(all(size(opts.u_lb)==[m,1]) && all(size(opts.u_ub)==[m,1]), 'u_lb/u_ub must be m×1');

    % ---------- QP setup ----------
    % Decision z = [theta; u] ∈ R^{n+m}
    I = eye(n);
    Z = zeros(n,m);

    % Equality: (I - A) theta - B u = 0  ->  [ (I-A)  -B ] z = 0
    Aeq = [I - A, -B];
    beq = zeros(n,1);

    % Bounds:
    lbz = [lb; opts.u_lb];
    ubz = [ub; opts.u_ub];

    % Objective: (theta - X_ref)' W (theta - X_ref) + lambda_u * u'u
    % -> z' H z + f' z + const
    W  = (opts.W + opts.W')/2;     % symmetrize
    H  = blkdiag(2*W, 2*opts.lambda_u*eye(m));
    f  = [-2*W*X_ref; zeros(m,1)];

    % ---------- solve ----------
    info = struct('exitflag', NaN, 'solver', '', 'message', '');
    switch lower(opts.solver)
        case 'quadprog'
            qp_opts = optimoptions('quadprog','Display','none');
            [z,~,exitflag,output] = quadprog(H, f, [], [], Aeq, beq, lbz, ubz, [], qp_opts);
            info.exitflag = exitflag; info.solver = 'quadprog'; info.message = output.message;
        case 'fmincon'
            % Convert to fmincon (same H,f with equality and bounds)
            fun = @(z) 0.5*z.'*H*z + f.'*z;
            nonlcon = [];
            x0 = [min(max(X_ref,lb),ub); zeros(m,1)]; % feasible w.r.t bounds (not equality)
            Aeq_fc = Aeq; beq_fc = beq;
            fm_opts = optimoptions('fmincon','Display','none','Algorithm','interior-point',...
                                   'SpecifyObjectiveGradient',false);
            [z,~,exitflag,output] = fmincon(fun, x0, [], [], Aeq_fc, beq_fc, lbz, ubz, nonlcon, fm_opts);
            info.exitflag = exitflag; info.solver = 'fmincon'; info.message = output.message;
        otherwise
            error('Unknown solver: %s', opts.solver);
    end

    if info.exitflag <= 0
        theta = nan(n,1); u = nan(m,1);
        warning('Equilibrium QP failed: %s', info.message);
        return;
    end

    theta = z(1:n);
    u     = z(n+1:end);
end
