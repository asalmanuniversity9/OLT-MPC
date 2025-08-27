function K = mykernel(XX, R, Kinfo)
    % MYKERNEL   Squared-exponential kernel matrix
    %
    %   K = mykernel(XX, R, Kinfo)
    %
    %   Inputs:
    %     XX      D-by-nx matrix of D data samples (each row is an nx-vector)
    %     R       M-by-nx matrix of M inducing points
    %     Kinfo   struct with fields
    %               .Name              must be 'SquaredExponential'
    %               .KernelParameters  2×1 vector [ell; sigma]
    %
    %   Output:
    %     K       D-by-M kernel matrix
    
    %— validate inputs
    if nargin<3 || ~isstruct(Kinfo)
        error('Please supply a valid kernel‐info struct.');
    end
    if ~isfield(Kinfo,'Name') || ~strcmp(Kinfo.Name,'SquaredExponential')
        error('Kernel.Type must be ''SquaredExponential''.');
    end
    params = Kinfo.KernelParameters;
    if numel(params)~=2
        error('KernelParameters must be a 2×1 vector [ell; sigma].');
    end
    ell   = params(1);
    sigma = params(2);
    
    D = size(XX,1);
    %— compute squared distances: D2(i,j) = || XX(i,:) – R(j,:) ||^2
    %   fast vectorized form:
    if D == 1
       D2 = sum( bsxfun(@minus, R, XX).^2 , 2 ).';  
    else
        XX2 = sum(XX.^2, 2);       % D×1
        RR2 = sum(R.^2, 2)';       % 1×M
        D2  = bsxfun(@plus, XX2, RR2) - 2*(XX * R');  % D×M
    end
    %— squared-exponential kernel
    K = (sigma^2) * exp( - D2 / (2*ell^2) );
end
