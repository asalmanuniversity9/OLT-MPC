function [t0, x0, u0] = shift(T, t0, x0, u,f,n)
st = [x0(1:n)];
con = u(1,:)';
f_value = f(st,con);
st = (f_value);
x0_t = full(st);
x0 = [x0_t(1:n-1);x0_t(end)]; 
t0 = t0 + T;
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end