function K = mykernel_single(XX, R, Kinfo)
    % MYKERNEL_SINGLE: Squared-exponential kernel vector (CASADI/Matlab-safe)
    %   XX:    1 x n_x (row vector)
    %   R:     n x n_x (rows: inducing points)
    %   Kinfo: .Name = 'SquaredExponential'
    %          .KernelParameters = [ell; sigma]
    %   K:     1 x n

    % Ensure XX is row vector
    XX = XX(:)'; % 1 x n_x

    % Parameters
    params = Kinfo.KernelParameters;
    ell   = params(1);
    sigma = params(2);

    n = size(R, 1);
    % Expand XX to match R’s rows
    XX_mat = repmat(XX, n, 1);  % n x n_x

    % Compute squared distances
    D2 = sum((R - XX_mat).^2, 2); % n x 1

    % Kernel vector (1 x n)
    K = (sigma^2) * exp( - D2' / (2*ell^2) );
end
