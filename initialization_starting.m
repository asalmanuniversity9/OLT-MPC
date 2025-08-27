%% OLT-MPC depth-plane AUV – discrete LQR demo with constraint overlays
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
n_x = size(A_c,1);
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

f1 = @(x) -alpha_w*x(3)*abs(x(3)) - beta_z*x(1);
f2 = @(x) -alpha_q*x(4)*abs(x(4)) - beta_th*x(2);

%%
B = [-1.86e-7 0;
    8.04e-8 0;
    -7.59e-6 -1.11e-7;
    3.21e-6 0];

Bnorm = norm(B,2);           % 8.2440e-6

fmax = [1.3; 0.12585];       % as in (1)
Gamma = Bnorm * norm(fmax);  % 1.078e-5
disp(Gamma)
%% ---------- physical / operating envelopes ----------
z_min  =   0 ;    z_max  =  50;     % depth   [m]
th_lim = 0.17;                      % |theta| ≤ 10° ≈ 0.17 rad
w_lim  = 1.0 ;                      % |w|     ≤ 1   rad s⁻¹
q_lim  = 0.5 ;                      % |q|     ≤ 0.5 rad s⁻¹

%% ---------- Initial GP --------------------------------------------------------
n_R = 20;
noise_sigma = 5e-4;
[gprMdl,xtrain,ytrain] = GP_initial_estimate(B,g,false,false,n_R,noise_sigma);
save_induce_points_full(xtrain, ytrain, gprMdl, n_R, 'Induce_point_history')
function_counter = size(B,2);

ytrain_0 = ytrain;


i = 1;
initial_size  = 100;                         % needs fewer points
for i = 1: function_counter
    K_RR{i}  = mykernel(gprMdl{i}.X,gprMdl{i}.X,gprMdl{i}.KernelInformation);
    iK_RR{i} = inv(K_RR{i});
end
dbclear if naninf
lhs = lhsdesign(initial_size,4,'criterion','maximin');
z     =  z_min  + lhs(:,1)*(z_max-z_min);
theta = -th_lim + lhs(:,2)*2*th_lim;
w     = -w_lim  + lhs(:,3)*2*w_lim;
q     = -q_lim  + lhs(:,4)*2*q_lim;
X_initial = [z theta w q];


%noise_t = noise_sigma*[randn(1,length(X_initial)); randn(1,length(X_initial))];

noise_t = truncated_gaussian_noise([2,length(X_initial)], noise_sigma, 0.1*noise_sigma);
Ytrue = zeros(initial_size,2);


K_XR = cell(function_counter,1);
H = cell(function_counter,1);
y_init = zeros(initial_size,2);
for i = 1:function_counter
    K_XR{i} = mykernel(X_initial,gprMdl{i}.X,gprMdl{i}.KernelInformation);
    H{i}    = K_XR{i}*iK_RR{i};
    %y_init(:,i)  = H{i}*(ytrain(:,i));
    %ytemp = sin(x_test(i));
end
for n = 1:initial_size
    Ytrue(n,:) = (pinv(B)*g(X_initial(n,:)'))+noise_t(:,n);     % the nonlinear term
end

sigmaF0 = std(ytrain);
%sigmaF0 = [ noise_sigma noise_sigma];
Sigma_t = K_RR;
ksi =  zeros(2,initial_size);
sigma_t_n = zeros(2,initial_size);
H_temp = cell(1,2);
S_t    = cell(1,2);
G_t    = cell(1,2);

for i = 1:initial_size
    
    for j = 1 : function_counter
        H_temp{j} = H{j}(i,:);
        S_t{j} = H_temp{j}* Sigma_t{j} * H_temp{j}' + sigmaF0(j) * eye(1);
        G_t{j} =  Sigma_t{j} * H_temp{j}'*inv(S_t{j});
        sigma_t_n(j,i) = norm(Sigma_t{j});
        
        isPosSemiDef = all(eig(Sigma_t{j}) >= 0);
        if (~isPosSemiDef)
           print("error") 
        end
        
        y_init(:,j)  = H{j}*(ytrain(:,j));
        
        ksi(j,i) =  Ytrue(i,j)-y_init(i,j);
        ytrain(:,j) =G_t{j}*ksi(j,i)+ ytrain(:,j);
        Sigma_t{j} = Sigma_t{j} -  G_t{j}*S_t{j}*G_t{j}';
        
    end
end

%% 3.  scatter (truth vs GP) for each output

test_size = 3000;
lhs = lhsdesign(test_size,4,'criterion','maximin');
z     =  z_min  + lhs(:,1)*(z_max-z_min);
theta = -th_lim + lhs(:,2)*2*th_lim;
w     = -w_lim  + lhs(:,3)*2*w_lim;
q     = -q_lim  + lhs(:,4)*2*q_lim;
X_test = [z theta w q];

K_XR_test = cell(function_counter,1);
H_test = cell(function_counter,1);

Ytrue_test = zeros(test_size,2);
y_test_plot = zeros(test_size,2);
y_test_plot_0 = zeros(test_size,2);


for n = 1:test_size
    Ytrue_test(n,:) = pinv(B)*g(X_test(n,:)');     % the nonlinear term
end
for i = 1:function_counter
    K_XR_test{i} = mykernel(X_test,gprMdl{i}.X,gprMdl{i}.KernelInformation);
    H_test{i}    = K_XR_test{i}*iK_RR{i};
    y_test_plot(:,i)  = H_test{i}*(ytrain(:,i));
    y_test_plot_0(:,i)  = H_test{i}*(ytrain_0(:,i));
    
    %ytemp = sin(x_test(i));
end

% if(1)
%     figure('Name','GP vs Truth — all four outputs','Position',[100 100 900 700])
%     titles = {'g₁(x)','g₂(x)','g₃(x)','g₄(x)'};
%     
%     for p = 1:2
%         subplot(2,1,p)
%         scatter(Ytrue(:,p), y_init(:,p), 14,'filled'); hold on
%         xy = [min(Ytrue(:,p)) max(Ytrue(:,p))];
%         plot(xy,xy,'k--','LineWidth',1);                        % 45° reference
%         xlabel('Truth  g_i(x)');  ylabel('GP prediction \mu_i');
%         title(titles{p});
%         grid on
%     end
%     sgtitle('GP mean prediction vs ground-truth nonlinear term')
% end

[n, function_counter] = size(Ytrue_test);
var_titles = arrayfun(@(k) sprintf('Output %d: g_{%d}(x)', k, k), 1:function_counter, 'UniformOutput', false);

figure('Name','Initial vs. Learned Model Predictions','Position',[100 100 900 700]);

for p = 1:function_counter
    % Row 1: Initial Model
    subplot(2, function_counter, p)
    scatter(Ytrue_test(:,p), y_test_plot_0(:,p), 14, 'filled');hold on
    xy = [min(Ytrue_test(:,p)), max(Ytrue_test(:,p))];
    plot(xy, xy, 'k--', 'LineWidth', 1); 
    xlabel('Ground Truth $g_i(x)$', 'Interpreter', 'latex');
    ylabel('Initial Prediction', 'Interpreter', 'latex');
    %title(['Initial: ', var_titles{p}], 'Interpreter', 'latex');
    grid on

    % Row 2: GP Model
    subplot(2, function_counter, function_counter + p)
    scatter(Ytrue_test(:,p), y_test_plot(:,p), 14, 'filled');hold on
    xy = [min(Ytrue_test(:,p)), max(Ytrue_test(:,p))];
    plot(xy, xy, 'k--', 'LineWidth', 1); 
    xlabel('Ground Truth $g_i(x)$', 'Interpreter', 'latex');
    ylabel('GP Prediction $\mu_i$', 'Interpreter', 'latex');
    %title(['GP: ', var_titles{p}], 'Interpreter', 'latex');
    grid on
end

sgtitle('Comparison: Initial vs. Learned GP Prediction for All Outputs', 'Interpreter', 'latex');

% Calculate RMSE for each output
rmse_init = sqrt(mean((y_test_plot_0 - Ytrue_test).^2, 1));
rmse_gp   = sqrt(mean((y_test_plot   - Ytrue_test).^2, 1));
improvement = 100 * (rmse_init - rmse_gp) ./ rmse_init;

% Set field widths for each column
fprintf('%-8s %-15s %-15s %-15s\n', 'Output', 'Initial RMSE', 'GP RMSE', 'Improvement (%)');
fprintf('%s\n', repmat('-',1,55));
for p = 1:function_counter
    fprintf('%-8d %-15.4f %-15.4f %-15.2f\n', p, rmse_init(p), rmse_gp(p), improvement(p));
end




[n_funcs, len] = size(sigma_t_n);
nrows = ceil(n_funcs/2);
ncols = min(n_funcs, 2); % Or choose another arrangement if you want

figure('Name','Norm of Predictive Variance for Each Output','Position',[100 100 900 350*nrows]);
titles = arrayfun(@(k) sprintf('Variance Norm: $g_{%d}(x)$', k), 1:n_funcs, 'UniformOutput', false);

for p = 1:n_funcs
    subplot(nrows, ncols, p)
    plot(1:len, sigma_t_n(p,:), 'LineWidth', 1.4);
    xlabel('Sample Index');
    ylabel('Variance Norm', 'Interpreter','latex');
    title(titles{p}, 'Interpreter','latex');
    grid on
end

sgtitle('Norm of Predictive Variance Across Samples for Each Output', 'Interpreter','latex');

%% ----------------------------------------- Tube modeling ---------------------------------------------
%% Calculate C1 and C2
C2 = 3*noise_sigma*sqrt(gprMdl{1}.KernelInformation.KernelParameters(2)*inv(noise_sigma));

%% Calculate max ||H||
[maxNorm, xStar] = maxKernelNorm_bounded(xtrain', gprMdl{1}.KernelInformation.KernelParameters(1), gprMdl{1}.KernelInformation.KernelParameters(2),[z_min -th_lim -w_lim -q_lim],[z_max th_lim w_lim q_lim]);
C1      = maxNorm^2;
C1_temp = n_R*(gprMdl{1}.KernelInformation.KernelParameters(2))^4;
%%
disaccount = 0.2;
[x_max_1, f1_max, exitflag1, ~] = maximize_on_compact_set(f1, [z_min -th_lim -w_lim -q_lim], [z_max th_lim w_lim q_lim], [0 0 0 0]);
[x_max_2, f2_max, exitflag2, ~] = maximize_on_compact_set(f2, [z_min -th_lim -w_lim -q_lim], [z_max th_lim w_lim q_lim], [0 0 0 0]);
[rkhs_ub_1, Bdelta_1] = gpr_rkhs_upper_bound(gprMdl{1}, 1e-10);
[rkhs_ub_2, Bdelta_2] = gpr_rkhs_upper_bound(gprMdl{2}, 1e-10);

Gamma_1 = max([(1+disaccount)*f1_max ,  Bdelta_1]);
Gamma_2 = max([(1+disaccount)*f2_max ,  Bdelta_2]);
%%

Gamma = sqrt(Gamma_1^2 + Gamma_2^2);
Cq_final = C1*Gamma

if Cq_final < 100
save_initial_structure()
end