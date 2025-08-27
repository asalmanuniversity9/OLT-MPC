function [P_t,E] = Pmaker(A,Q,B,K,factor)

A_cl = A-B*K;
Q_h = Q*Q;
P_h  = dlyap(A_cl', Q_h,[], sqrt(factor)*eye(4));
P_t = chol(P_h);
E = P_t*A_cl*inv(P_t);
%ss = max(svd(A_cl));
%[V,D] = eig(A_cl)
%lambda = max(diag(abs(D)))
%beta_t = cond(V)^2*(lambda^2)/(1-lambda^2)
%Cond(Q) = 1/ss^2/(1-ss^2)/(1+beta_t)
end