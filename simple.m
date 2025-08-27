%% Dive-plane model for REMUS-class AUV  (Hewing et al., TCST 2020 §IV)
%   x = [theta; w; q]
%   u = delta_s   (stern-plane deflection, rad)

clear;  clc;

%% 1.  Hydrodynamic parameters (continuous-time)
a22 =  0.95;   a23 =  0.46;
a32 = -0.85;   a33 = -2.20;
b2  =  0.40;   b3  =  3.50;

%% 2.  Non-linear residuals (ground truth for simulation)
g1 = @(w,q) -0.8*w.*abs(w)  - 0.15*q.*abs(q) - 0.10*w.*q;  % [m/s²]
g2 = @(w,q) -1.5*w.*abs(w)  - 0.40*q.*abs(q) - 0.20*w.*q;  % [rad/s²]

%% 3.  Continuous-time linear model (small-signal)
Ac = [ 0     0     1  ;
       0  a22  a23  ;
       0  a32  a33 ];
Bc = [ 0 ; b2 ; b3 ];

%% 4.  Discretise with ZOH
Ts = 0.1;                              % [s]
sysC   = ss(Ac,Bc,[],[]);
sysD   = c2d(sysC,Ts,'zoh');
Ad = sysD.A;  Bd = sysD.B;

disp('Discrete A and B:');  disp(Ad);  disp(Bd);

%% 5.  LQR-based ancillary stabiliser  u = -Kx
Q = diag([ 1  0  10]);     % pitch, w, q
R = 20;
[K,~,eigCL] = dlqr(Ad,Bd,Q,R);
fprintf('Closed-loop poles  =  [%s]\n', sprintf('%.3f ',eigCL))

%% 6.  Simulation of the NON-linear truth + LQR
Tf     = 20;                     % 20 s
Nsteps = Tf/Ts;
x      = zeros(3,Nsteps+1);      % pre-allocate state trajectory
u      = zeros(1,Nsteps);
ref    = [deg2rad(30);0;0];      % command 30 deg pitch (step)

for k = 1:Nsteps
    % LQR around zero trim (no integral action here)
    u(k) = -K*(x(:,k) - ref);
    
    % Saturate stern-plane to ±20 deg
    u(k) = max(min(u(k), deg2rad(20)), -deg2rad(20));
    
    % -----  truth dynamics (forward Euler for brevity) ----------
    theta = x(1,k);     w = x(2,k);    q = x(3,k);
    
    dtheta = q;
    dw     = a22*w + a23*q + b2*u(k) + g1(w,q);
    dq     = a32*w + a33*q + b3*u(k) + g2(w,q);
    
    x(:,k+1) = x(:,k) + Ts*[dtheta; dw; dq];
end

t = 0:Ts:Tf;

%% 7.  Plots
figure;
subplot(3,1,1);
plot(t, rad2deg(x(1,:)),'LineWidth',1.2); hold on;
yline(30,'--','Ref 30°');  ylim([0 35]);
ylabel('\theta [deg]'); grid on;

subplot(3,1,2);
plot(t, x(2,:),'LineWidth',1.2);  ylabel('w [m/s]'); grid on;

subplot(3,1,3);
stairs(t(1:end-1), rad2deg(u),'LineWidth',1.2);
ylabel('\delta_s [deg]'); xlabel('time [s]'); grid on;
