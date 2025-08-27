function [normH, alpha] = gpRKHSNorm(gprMdl)
%GP_RKHSNORM  RKHS norm of the GP posterior mean stored in gprMdl
%
%   normH  – scalar RKHS norm  ||m_D||_H
%   alpha  – (optional) alpha weights = (K + σ_n^2 I)^{-1} y

    alpha  = gprMdl.Alpha;   % already (K + σ_n^2 I)^{-1} y
    normH2 = gprMdl.Y' * alpha;
    normH  = sqrt(normH2);
end
