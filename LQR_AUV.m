%% BB2 depth-plane AUV – discrete LQR demo with constraint overlays
clc; clear; close all;

%% 1. Continuous-time matrices (paper, Eqs 44–45)  … unchanged …
A_c = [ 9.056e-14  1.999       1          -9.466e-12 ;
       -1.527e-13  1.59e-14   -2.135e-11   1          ;
        1.300e-04 -3.04e-07  -0.036       -1.144      ;
        7.005e-07 -0.0149     -0.00258    -0.095     ];
B_c = [  3.257e-14  -3.078e-19 ;
        -9.036e-14  -1.156e-18 ;
        -1.5e-04    -2.219e-06 ;
         6.444e-05   4.595e-17 ];

T_s  = 0.05;                           % sample time [s]

%% 2. ZOH discretisation
sys_d = c2d(ss(A_c,B_c,eye(4),zeros(4,2)), T_s, 'zoh');
A = sys_d.A;  B = sys_d.B;

%% 3. Discrete LQR weights
Q = 200*diag([50 50 50 50]);
R = diag([10 10]);
K = dlqr(A,B,Q,R);

%% 4. Disturbance / g(x)  (illustrative)
alpha_w = 0.8;  alpha_q = 0.5;
beta_z  = 0.01; beta_th = 0.005;
f = @(x) [ -alpha_w*x(3)*abs(x(3)) - beta_z*x(1) ; ...
           -alpha_q*x(4)*abs(x(4)) - beta_th*x(2) ];
g = @(x) B*f(x);
%%
B = [-1.86e-7 0;
      8.04e-8 0;
     -7.59e-6 -1.11e-7;
      3.21e-6 0];

Bnorm = norm(B,2);           % 8.2440e-6

fmax = [1.3; 0.12585];       % as in (1)
Gamma = Bnorm * norm(fmax);  % 1.078e-5
disp(Gamma)

%% 5.  Simulation parameters
Tf   = 3000;                      N = Tf/T_s;
x    = zeros(4,N+1);
u    = zeros(2,N);
zRef = [15*ones(1,N/4)  20*ones(1,3*N/4)];  % depth step
thRef= zeros(1,N);

x(:,1) = [15; 0.05; 0; 0];                 % initial state

%% --- constraints -------------------------------------------------------
z_min =   0;      z_max = 50;              % depth envelope [m]
th_lim =  0.17;                            % |theta| ≤ 10°  ≈ 0.17 rad
dv_lim =  0.35;                            % |δ_v| ≤ 20°    ≈ 0.35 rad
dm_lim = 50;                               % |δ_m| ≤ 50 kg
% ------------------------------------------------------------------------

%% 6. Closed-loop simulation
for k = 1:N
    x_ref = [zRef(k); thRef(k); 0; 0];
    u(:,k) = -K * (x(:,k) - x_ref);
    x(:,k+1) = A*x(:,k) + B*u(:,k) + g(x(:,k));
end
t = 0:T_s:Tf;

%% 7. Plot with constraint overlays
figure;

% --- depth --------------------------------------------------------------
subplot(3,1,1)
plot(t, x(1,:), 'LineWidth',1.3); hold on
plot(t(1:end-1), zRef, '--', 'LineWidth',1.0)
% plotting constraints
yline(z_min, ':', 'Depth min',  'LabelHorizontalAlignment','left');
yline(z_max, ':', 'Depth max',  'LabelHorizontalAlignment','left');
ylabel('Depth  z [m]')
legend('z(t)','z_{ref}','Location','best'); grid on

% --- pitch --------------------------------------------------------------
subplot(3,1,2)
plot(t, x(2,:), 'LineWidth',1.3); hold on
yline( th_lim, ':', '+θ_{lim}');
yline(-th_lim, ':', '-θ_{lim}');
ylabel('Pitch  θ [rad]'); grid on
legend('θ(t)','Location','best')

% --- control inputs -----------------------------------------------------
subplot(3,1,3)
stairs(t(1:end-1), u(1,:), 'LineWidth',1.3); hold on
stairs(t(1:end-1), u(2,:), 'LineWidth',1.3)
% plotting constraints
yline( dv_lim, '--', '+δ_v^{lim}');
yline(-dv_lim, '--', '-δ_v^{lim}');
yline( dm_lim, '--', '+δ_m^{lim}');
yline(-dm_lim, '--', '-δ_m^{lim}');
ylabel('Inputs'); xlabel('Time  [s]'); grid on
legend('δ_v','δ_m','Location','best')

sgtitle('BB2 discrete LQR - with constraint overlays')
