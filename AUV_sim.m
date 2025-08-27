%% OLT-MPC depth-plane AUV – discrete OLT-MPC with constraint
clc; clear; close all;
%%
addpath('Casadi')
import casadi.*
%%
function_counter = 2;
result_Design_P = 0;
initdata = load_initial_structure_by_name('IS_1p95e+01_2025-07-22_151058.mat');
data_z = initdata;
%%
noise_sigma = 5e-4;
Q = 10*eye(4);
R = 10*eye(2);
A = initdata.A;
A(2:4) = 0;
B = initdata.B;
xtrain = initdata.xtrain;
ytrain = initdata.ytrain;
n_d = 100;
%%
% Compute quantities
eig_A = eig(A).';
%K = dlqr(A,B,Q,R);

%K = place(A,B,[0.91 0.93 0.99 0.99])
%K = place(A,B,[0.91 0.98 0.94 0.96]);
K = place(A,B,[0.94 0.96 0.98 0.98]);

Acl = A - B*K;
eig_Acl = eig(Acl).';
rho_n = 0.01;
rho_l = 0.015;

%P_n = solve_scaled_lyap(Acl,rho_n);
%P_n = solve_diag_lyap_min(Acl,rho_n);
P_n = dlyap(Acl', 1e-9*eye(4),[], sqrt(1-rho_n)*eye(4));
%P_n = solve_scaled_lyap_diag(Acl,rho_n);

eig_Pn = eig(P_n).';

%P_l = solve_scaled_lyap(Acl,rho_l);
%P_l = solve_diag_lyap_min(Acl,rho_l);
P_l = dlyap(Acl', 1e-9*eye(4),[], sqrt(1-rho_l)*eye(4));
%P_l = solve_scaled_lyap_diag(Acl,rho_l);

eig_Pl = eig(P_l).';

gamma_n = norm(P_n);
gamma_l = norm(P_l);


if result_Design_P ==  1
    % Prepare row labels
    ResultNames = {
        'Open-loop Eig(A)';
        'Closed-loop Eig(A-BK)';
        'Eig(P_n), Lyap eps=1e-4';
        'Eig(P_l), Lyap eps=1e-3';
        'LQR Gain K (row 1)';
        'LQR Gain K (row 2)';
        };
    
    % Compose results matrix
    ResultsMatrix = [
        eig_A;
        eig_Acl;
        eig_Pn;
        eig_Pl;
        K(1,:);
        K(2,:)
        ];
    
    % Create table
    ResultsTable = array2table(ResultsMatrix, ...
        'VariableNames', {'Val1','Val2','Val3','Val4'}, ...
        'RowNames', ResultNames);
    
    disp('Summary Table of Initialization Results:');
    disp(ResultsTable)
end
%%
ub = [initdata.z_min initdata.th_lim initdata.w_lim initdata.q_lim];
lb = [-initdata.z_max -initdata.th_lim -initdata.w_lim -initdata.q_lim];
[lb_tight, ub_tight, psi,x_tube_n,nu_n_0] = tighten_bounds_by_tube(lb, ub,...
    initdata.C1, [ytrain(:,1);ytrain(:,2)] ,...
    initdata.Gamma, blkdiag(initdata.Sigma_t{1},initdata.Sigma_t{2}),...
    initdata.C2, n_d, P_n,norm(B,2),gamma_n,rho_n^-1);
[lb_tight_l, ub_tight_l, psi_l,x_tube_l,nu_l_0] = tighten_bounds_by_tube(lb, ub,...
    initdata.C1, [ytrain(:,1);ytrain(:,2)] ,...
    initdata.Gamma, blkdiag(initdata.Sigma_t{1},initdata.Sigma_t{2}),...
    initdata.C2, n_d, P_l,norm(B,2),gamma_l,rho_l^-1);

disp('Tightened bounds (nominal):');
fprintf('  lb_tight   = [ %s ]\n', num2str(lb_tight, '%.4f '));
fprintf('  lb_tight_l = [ %s ]\n', num2str(lb_tight_l, '%.4f '));

disp('Tightened bounds (learned):');
fprintf('  ub_tight   = [ %s ]\n', num2str(ub_tight, '%.4f '));
fprintf('  ub_tight_l = [ %s ]\n', num2str(ub_tight_l, '%.4f '));
%%

dt = 0.05;
X_ref =[-20 0 0 0];
K_tube = K;
T = dt; %[s]
N = 20; % prediction horizon

%%
x = MX.sym('x',[8,1]);
states = [x]; n_states = length(states);
v = MX.sym('v',[2,1]); % u1~2
controls = [v]; n_controls = length(controls);
av = MX.sym('av',[2,1]); % u1~2
a_controls = [av]; an_controls = length(a_controls);
act_x = MX.sym('act_x',[4,1]);
act_states = [act_x]; act_n_states = length(act_states);


mu = MX.sym('mu',[length(xtrain),2]);
mu_t = [mu]; mu_states = length(mu_t);
%%
A_s = [Acl , B*K; zeros(4,4) , A];
B_s = [B;B];
B_l = [B;zeros(n_states/2,n_controls)];
%%
for i = 1: function_counter
    K_RR{i}  = mykernel(initdata.gprMdl{i}.X,initdata.gprMdl{i}.X,initdata.gprMdl{i}.KernelInformation);
    iK_RR{i} = inv(K_RR{i});
end
%%
f_exact = initdata.f;
rhs = A_s*states+B_s*controls+B_l*...
    [mykernel_single(states(1:4)',initdata.gprMdl{1}.X,initdata.gprMdl{1}.KernelInformation)*iK_RR{1}*(mu_t(:,1));...
    mykernel_single(states(1:4)',initdata.gprMdl{2}.X,initdata.gprMdl{2}.KernelInformation)*iK_RR{2}*(mu_t(:,2))]; % system r.h.s
rhs_actual = A*act_x+B*a_controls+B*f_exact(act_x); % system r.h.s
% %rhs_actual = A_actual*act_x+B_actual*a_controls; % system r.h.s

f_nominal = Function('f',{states,controls,mu_t},{rhs}); % nonlinear mapping function f(x,u)
f_actual  = Function('f_act',{act_x,a_controls},{rhs_actual});
%%
U = MX.sym('U',n_controls,N); % Decision variables (controls)
X = MX.sym('X',n_states,(N+1));
P = MX.sym('P',n_states + N*(n_states+n_controls)+n_states+2*length(xtrain)); % initial condition + MPC+final state+ Pv
%%
obj = 0; % Objective function
g = [];  % constraints vector

Q_n = 10*eye(n_states/2,n_states/2); % weighing matrices (states)
Q = blkdiag(Q_n,zeros(n_states/2,n_states/2));

Rp = 1*eye(n_controls,n_controls); % weighing matrices (controls)

%K_p = place(A_p,B_actual,0.99*ones((length(R)+1),1));
factor = 0.97;
[P_t,E] = Pmaker(A,Q_n,B,K,factor);
ito = (1-norm(E));
kappa = norm(Q_n*inv(P_t))+norm(Rp*K*inv(P_t));
P_s_t = 1*kappa/ito* P_t;
P_s = blkdiag(P_s_t,zeros(4,4));
%%
st  = X(:,1); % initial state
%g   = [g;st-P(1:n_states)]; % initial condition constraints
%g   = [g;(st(1:4)-P(1:4))'*(st(1:4)-P(1:4));(st(5:n_states)-P(5:n_states))'*(st(5:n_states)-P(5:n_states))]; % initial condition constraints
g   = [g;(st(1:4)-P(1:4));(st(5:n_states)-P(5:n_states))]; % initial condition constraints

g_t = [];
g_f = [];
%%
count_t = n_states+n_controls;
start_t = n_controls - 1;
end_t   = n_states - start_t-1;
%%
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(((st-P(count_t*k-start_t:count_t*k+end_t))'*(Q'*Q)*(st-P(count_t*k-start_t:count_t*k+end_t)))) + ...
        ((con-P(count_t*k+(end_t+1):count_t*k+(end_t+n_controls)))'*(Rp'*Rp)*(con-P(count_t*k+(end_t+1):count_t*k+(end_t+n_controls)))) ; % calculate obj
    st_next = X(:,k+1);
    f_value = f_nominal(st,con,...
        [P(count_t*(N+1)+end_t+1:(count_t*(N+1)+end_t+length(xtrain))),P((count_t*(N+1)+end_t+length(xtrain)+1):end)]);
    st_next_euler = (f_value);
    g   = [g;st_next-st_next_euler]; % compute constraints
end
%%
k = k+1;
obj = obj+((st_next-P(count_t*k-start_t:count_t*k+end_t))'*(P_s'*P_s)*(st_next-P(count_t*k-start_t:count_t*k+end_t)));
g = [g;g_f;g_t];
OPT_variables = [reshape(X,n_states*(N+1),1);reshape(U,n_controls*N,1)];
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);
%%
opts = struct;
opts.ipopt.max_iter = 1000;
opts.ipopt.print_level =1;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-10;
%opts.ipopt.constr_viol_tol = 1e-20;
opts.ipopt.honor_original_bounds = 'no';
opts.ipopt.acceptable_obj_change_tol = 1e-20;
solver = nlpsol('solver', 'ipopt', nlp_prob,opts);
%%
args = struct;
%%
% args.lbg(1:(2+n_states*(N))) = 0;
% args.ubg(1:(2+n_states*(N))) = 0;

args.lbg(1:(8+n_states*(N))) = 0;
args.ubg(1:(8+n_states*(N))) = 0;
%%

% delta_f_vect = [x_tube_b/5;x_tube_b];
args.lbg(1:8) = -[(x_tube_l).^2,(x_tube_n).^2];
args.ubg(1:8) =  [(x_tube_l).^2,(x_tube_n).^2];
%%


args.lbx(1:n_states:n_states*(N+1),1) =  -inf ;
args.ubx(1:n_states:n_states*(N+1),1) =  +inf ;
args.lbx(2:n_states:n_states*(N+1),1) =  -inf ;
args.ubx(2:n_states:n_states*(N+1),1) =  +inf ;
args.lbx(3:n_states:n_states*(N+1),1) =  -inf ;
args.ubx(3:n_states:n_states*(N+1),1) =  +inf ;
args.lbx(4:n_states:n_states*(N+1),1) =  -inf ;
args.ubx(4:n_states:n_states*(N+1),1) =  +inf ;

args.lbx(5:n_states:n_states*(N+1),1) = lb_tight(1)  ;
args.ubx(5:n_states:n_states*(N+1),1) = ub_tight(1)  ;
args.lbx(6:n_states:n_states*(N+1),1) = lb_tight(2)  ;
args.ubx(6:n_states:n_states*(N+1),1) = ub_tight(2)  ;
args.lbx(7:n_states:n_states*(N+1),1) = lb_tight(3)  ;
args.ubx(7:n_states:n_states*(N+1),1) = ub_tight(3)  ;
args.lbx(8:n_states:n_states*(N+1),1) = lb_tight(4)  ;
args.ubx(8:n_states:n_states*(N+1),1) = ub_tight(4)  ;
%%
delta_u = abs(K*x_tube_n);
u_lim = [0.7 4000]';
%u_lim = [0.7 120]';
u_ub_tigh = u_lim-delta_u;
args.lbx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(1)+delta_u(1);
args.ubx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(1)-delta_u(1);
args.lbx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(2)+delta_u(2);
args.ubx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(2)-delta_u(2);
%% Checking equilibrium function
%
% opts = struct('u_lb', -[0.1 inf]', 'u_ub', [0.1 inf]', 'lambda_u', 0, 'W', [1 0 0 0; 0 0 0 0; 0 0 0 0 ; 0 0 0 0]);
% [theta_no, u_no, ~] = find_equilibrium_ss(A, B, -X_ref', lb_tight', ub_tight', opts);
%
% %%
% optsgp = struct;
% optsgp.idx_g = [3;4];              % map GP outputs into rows 3 and 4 of the state equation
% optsgp.x0    = theta_no;              % start at nominal
% optsgp.solver = 'fsolve';          % or 'lsqnonlin' if you also pass lb/ub
% optsgp.Display = 'off';
% [theta_s, ~] = solve_equilibrium_gp(Acl, B, theta_no, ytrain, initdata, iK_RR, optsgp);

optseq = struct('u_lb', -u_ub_tigh, ...
    'u_ub',  u_ub_tigh, ...
    'lambda_u', 0, ...
    'W', [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0], ...
    'idx_g', [3;4], ...
    'solver','fsolve', ...
    'Display','off');

[theta_s, theta_no, u_s, u_no] = Equilibrium_estimator( ...
    A, B, K, X_ref, lb_tight, ub_tight, initdata, iK_RR, ytrain, optseq);


%%
t0 = 0;
x0 = [-15;0;0;0;-15;0;0;0];    % initial condition.
xs = X_ref'; % Reference posture.


xx(:,1) = x0; % xx contains the history of states
t(1) = t0;
u0 = zeros(1,n_controls);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables
U0 = repmat(u0,1,N)';

sim_tim = 500; % Maximum simulation time
mpciter     = 0;
xx1         = [];
u_cl        = [];
u_act_cl    = [];
J_t         = [];

iter_numer = sim_tim/dt;
theta_s_ref = zeros(iter_numer,act_n_states);
u_s_ref = zeros(iter_numer,n_controls);
theta_mu_ref = zeros(iter_numer,act_n_states);
u_mu_ref = zeros(iter_numer,n_controls);
x_ref_r = zeros(iter_numer,act_n_states);
ss = 0;
ref_1 = [-20 0 0 0];
ref_2 = [-15 0 0 0];
ref_3 = [-20 0 0 0];

%%
for ref_itr = 1:iter_numer
    %x_ref_r(ref_itr,:) = ref_1;
    if (ref_itr*dt < 130)
        x_ref_r(ref_itr,:) = ref_1;
    elseif (ref_itr*dt >= 130) && (ref_itr*dt <350)
        x_ref_r(ref_itr,:) = ref_2;
    elseif (ref_itr*dt >= 350) && (ref_itr*dt <500)
        x_ref_r(ref_itr,:) = ref_3;
    else
        x_ref_r(ref_itr,:) = [0 0 0 0];
    end
    [theta_s_ref(ref_itr,:),theta_mu_ref(ref_itr,:),u_mu_ref(ref_itr,:),u_s_ref(ref_itr,:)]=...
        Equilibrium_estimator( ...
        A, B, K,...
        x_ref_r(ref_itr,:),...
        lb_tight, ub_tight, ...
        initdata,...
        iK_RR, ytrain,...
        optseq);
end
%%
initial_point = x0(1:4);

ygp_temp = ytrain;


main_loop = tic;
sigma_t_n = zeros(2,sim_tim / dt);
H_temp = cell(1,2);
S_t    = cell(1,2);
G_t    = cell(1,2);
while(mpciter < sim_tim / dt)
    
    X_ref = theta_mu_ref(mpciter+1,:);
    args.p(1:n_states) = [x0];
    
    
    
    args.lbx(5:n_states:n_states*(N+1),1) = lb_tight(1)  ;
    args.ubx(5:n_states:n_states*(N+1),1) = ub_tight(1)  ;
    args.lbx(6:n_states:n_states*(N+1),1) = lb_tight(2)  ;
    args.ubx(6:n_states:n_states*(N+1),1) = ub_tight(2)  ;
    args.lbx(7:n_states:n_states*(N+1),1) = lb_tight(3)  ;
    args.ubx(7:n_states:n_states*(N+1),1) = ub_tight(3)  ;
    args.lbx(8:n_states:n_states*(N+1),1) = lb_tight(4)  ;
    args.ubx(8:n_states:n_states*(N+1),1) = ub_tight(4)  ;
    % input discritize error violation
    delta_u = abs(K*x_tube_n);
    u_ub_tigh = u_lim-delta_u;
    args.lbx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(1)+delta_u(1);
    args.ubx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(1)-delta_u(1);
    args.lbx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(2)+delta_u(2);
    args.ubx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(2)-delta_u(2);
    
    for k = 1:N %new - set the reference to track
        if ((mpciter+1)*dt >= 50) && ((mpciter+1)*dt <80)
            ss = 0.1;
        elseif ((mpciter+1)*dt >= 80) && ((mpciter+1)*dt <90)
            %ss = 0;
        elseif ((mpciter+1)*dt >= 90) && ((mpciter+1)*dt <110)
            ss = 0;
        end
        args.p( count_t*k-start_t:count_t*k+end_t ) = [theta_mu_ref(mpciter+1,:)';theta_s_ref(mpciter+1,:)'];
        args.p( count_t*k+(end_t+1):count_t*k+(end_t+n_controls) ) = u_s_ref(mpciter+1,:);
    end
    k = k +1;
    args.p( count_t*k-start_t:count_t*k+end_t ) = [theta_mu_ref(mpciter+1,:)';theta_s_ref(mpciter+1,:)'];
    args.p(count_t*k+end_t+1:count_t*k+end_t+length(xtrain)) = ytrain(:,1);
    args.p(count_t*k+end_t+length(xtrain)+1:count_t*k+end_t+2*length(xtrain)) = ytrain(:,2);
    
    args.x0  = [reshape(X0',n_states*(N+1),1);reshape(U0',n_controls*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',[args.p]);
    u = reshape(full(sol.x(n_states*(N+1)+1:end))',n_controls,N)'; % get controls only from the solution
    xx1(:,1:n_states,mpciter+1)= reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
    
    %noise_t = truncated_gaussian_noise(1, 1e+5*noise_sigma, 3*1e+5*noise_sigma);
    %err_fdback = -K*(xx1(1,5:(n_states),mpciter+1)'-x0(5:(n_states)))+[0.005*noise_t;noise_t];
    err_fdback = -K*(xx1(1,5:(n_states),mpciter+1)'-x0(5:(n_states)));

    u_temp = repmat(err_fdback(1:n_controls)',N,1)+u(:,1:(n_controls));
    u_act = u_temp(1,:);
%     if ((abs(u_act(1)) > u_lim(1)) || (abs(u_act(2)) > u_lim(2)) )
%        %solver is violent constraint 
%        disp('solver is violent constraint')
%        if (abs(u_act(1)) > u_lim(1))
%            u_act(1) = sign(u_act(1))*u_lim(1);
%        end
%        if (abs(u_act(2)) > u_lim(2))
%            u_act(2) = sign(u_act(2))*u_lim(2);
%        end
%     end
    u_act_cl = [u_act_cl  ; u_act(1,:) ];
    x_t_0 = x0(1:n_states/2);
    u_t_0 = u_act';
    [t0, x0, u0] = shift(T, t0, x0, u_act,f_actual,act_n_states);
    noise_t_p = truncated_gaussian_noise([4 1],9e-1*noise_sigma, 3*9e-1*noise_sigma);
    x_t_1 = x0+noise_t_p;
    x0 = x0+noise_t_p;
    x0 = [x0;x0];
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)'; % get solution TRAJECTORY
    U0 =  reshape(full(sol.x(n_states*(N+1)+1:end))',n_controls,N)';
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:);X0(end,:)];
    U0 = [U0(2:end,:);U0(end,:)];
    
    %mpciter
    mpciter = mpciter + 1
    
    J_t = [J_t,sol.f];
    if(length(J_t)>10)
        if(full(J_t(end))> full(J_t(end-1)))
            if(abs(full(J_t(end))- full(J_t(end-10))) > 0.05)
                disp(J_t(length(J_t)-2:end))
            end
        end
    end
    noise_t = truncated_gaussian_noise(1, noise_sigma, 3*noise_sigma);
    Ytrue = (pinv(B)*(x_t_1 - A*x_t_0-B*u_t_0))+noise_t;
    for j = 1 : function_counter
        K_XR{j} = mykernel_single(x_t_0,initdata.gprMdl{j}.X,initdata.gprMdl{j}.KernelInformation);
        H{j}    = K_XR{j}*iK_RR{j};
        H_temp{j} = H{j};
        S_t{j} = H_temp{j}* initdata.Sigma_t{j} * H_temp{j}' + noise_sigma^2 * eye(1);
        G_t{j} =  initdata.Sigma_t{j} * H_temp{j}'*inv(S_t{j});
        %sigma_t_n(j,mpciter) = norm(Sigma_t{j});
        
        isPosSemiDef = all(eig(initdata.Sigma_t{j}) >= 0);
        if (~isPosSemiDef)
            print("error")
        end
        
        y_init(j)  = H{j}*(ytrain(:,j));
        
        ksi(j,i) =  Ytrue(j)-y_init(j);
        ytrain_temp(:,j) =G_t{j}*ksi(j,i)+ ytrain(:,j);
        Sigma_t{j} = initdata.Sigma_t{j} -  G_t{j}*S_t{j}*G_t{j}';
        
    end
    [lb_tight_t, ub_tight_t, psi_t,x_tube_n_t,nu_n] = tighten_bounds_by_tube(lb, ub,...
        initdata.C1, [ytrain_temp(:,1);ytrain_temp(:,2)] ,...
        initdata.Gamma, blkdiag(Sigma_t{1}, Sigma_t{2}),...
        initdata.C2, n_d+1, P_n,norm(B,2),gamma_n,rho_n^-1);
    if(nu_n < nu_n_0)
        disp('Update')
        nu_n_0 = nu_n;
        initdata.Sigma_t= Sigma_t;
        ytrain = ytrain_temp;
        initdata.G_t =  G_t;
        initdata.S_t =  S_t;
        initdata.sigma_t_n =  [initdata.sigma_t_n,[norm(Sigma_t{1}),norm(Sigma_t{2}) ]' ];
        lb_tight = lb_tight_t; ub_tight = ub_tight_t;
        delta_u = abs(K*x_tube_n_t);
        u_ub_tigh = u_lim-delta_u;
        args.lbx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(1)+delta_u(1);
        args.ubx(n_states*(N+1)+1:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(1)-delta_u(1);
        args.lbx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =  -u_lim(2)+delta_u(2);
        args.ubx(n_states*(N+1)+2:n_controls:n_states*(N+1)+n_controls*N,1) =   u_lim(2)-delta_u(2);
    end
    [lb_tight_l_t, ub_tight_l_t, psi_l_t,x_tube_l_t,nu_l] = tighten_bounds_by_tube(lb, ub,...
        initdata.C1, [ytrain_temp(:,1);ytrain_temp(:,2)] ,...
        initdata.Gamma, blkdiag(Sigma_t{1}, Sigma_t{2}),...
        initdata.C2, n_d+1, P_l,norm(B,2),gamma_l,rho_l^-1);
end

main_loop_time = toc(main_loop);
average_mpc_time = main_loop_time/(mpciter+1)
t_j_t = t;
t = [t,t(end)+dt];
%%
ExRSGP.t                  = sim_tim;
ExRSGP.init_data          = data_z;
ExRSGP.update_data        = initdata;
ExRSGP.x_ref_r            = x_ref_r;
ExRSGP.K                  = K;

ExRSGP.theta_mu_ref      = theta_mu_ref;

ExRSGP.xx                 = xx;
ExRSGP.u_cl               = u_cl;
ExRSGP.u_act_cl           = u_act_cl;
ExRSGP.average_mpc_time   = average_mpc_time;

%%
ExRSGP_20 = ExRSGP;
save('workspace\Explor_RSGP_result_20','ExRSGP_20')

%%
% ---------- publication-quality stacked plots ----------
fig = figure('Name','AUV States vs References', ...
    'Position',[100 100 950 900], ...
    'Renderer','painters');   % vector-friendly

% Global LaTeX interpreters (safe for this figure only)
set(fig, 'DefaultTextInterpreter','latex', ...
    'DefaultAxesTickLabelInterpreter','latex', ...
    'DefaultLegendInterpreter','latex');

t_p  = 0:dt:sim_tim;
idx  = 2:numel(t_p)-2;      % align with x_ref_r, theta_mu_ref (your sample used 2:end-2)
tref = t_p(idx);

state_names = {'$z_k$ [m]', '$\theta_k$ [rad]', '$w_k$ [m/s]', '$q_k$ [rad/s]'};

% Color/linestyle choices (consistent & colorblind-friendly)
clr_state = [0 0.45 0.74];      % blue
clr_ref   = [0.85 0.33 0.10];   % red-orange
ls_ref    = '--';
ls_theta  = '--';

tiledlayout(4,1,'Padding','compact','TileSpacing','compact');
ax = gobjects(1,4);

for i = 1:4
    ax(i) = nexttile; hold on; %grid on; grid minor;
    
    % State
    p1 = plot(t_p,              xx(i,:),           'LineWidth', 2.4, 'Color', clr_state);
    
    % Reference (matching your indexing)
    p2 = plot(tref,             x_ref_r(idx,i),    ls_ref, 'LineWidth', 2.0, 'Color', clr_ref);
    
    % GP mean
    p3 = plot(tref,             theta_mu_ref(idx,i),'k--',  'LineWidth', 2.0);
    
    % Y label
    ylabel(state_names{i}, 'FontSize', 16);
    
    % Only bottom subplot gets the x-label
    if i < 4
        %ax(i).XAxis.Visible = 'off';
    else
        xlabel('$t$ [s]','FontSize',16);
    end
    
    % Axis font
    set(ax(i), 'FontName','Times New Roman', 'FontSize', 14);
    
    % Optional: auto y-limits with small padding for clarity
    ydata_all = [xx(i,:), x_ref_r(idx,i).', theta_mu_ref(idx,i).'];
    ylo = min(ydata_all); yhi = max(ydata_all);
    pad = 0.06 * max(1e-12, yhi - ylo);
    ylim([ylo - pad, yhi + pad]);
    
    % Legend only once (top panel), compact
    if i == 1
        lg = legend([p1 p2 p3], {'State','Reference','$\theta_\mu$'}, ...
            'Location','best', 'FontSize', 12);
        %lg.Box = 'off';
    end
end

linkaxes(ax,'x');

%sgtitle('AUV State Trajectories vs References', 'FontSize', 18, 'FontWeight','normal');
%%

% ---------- Inputs with bounds (2×1) ----------
figU = figure('Name','AUV Inputs vs Bounds', ...
              'Position',[100 100 950 500], ...
              'Renderer','painters');

set(figU, 'DefaultTextInterpreter','latex', ...
          'DefaultAxesTickLabelInterpreter','latex', ...
          'DefaultLegendInterpreter','latex');

t_u = t_p(2:end);   % assumes inputs sampled at same dt

inp_names = {'$\delta_{v,k}$', '$\delta_{m,k}$'};

% Colors & styles
clr_u   = [0 0.45 0.74];    % actual input (blue)
clr_bnd = [0.3 0.3 0.3];    % original bounds (gray)
clr_tig = [0.85 0.33 0.10]; % tightened bounds (red-orange)
ls_bnd  = '-.';             % bound linestyle
ls_tig  = '--';             % tightened linestyle

tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
axu = gobjects(1,2);

inp_names = {'$\delta_{v,k}$', '$\delta_{m,k}$'};

% Colors & styles
clr_u   = [0 0.45 0.74];    % actual input (blue)
clr_bnd = [0.3 0.3 0.3];    % original bounds (gray)
clr_tig = [0.85 0.33 0.10]; % tightened bounds (red-orange)
ls_bnd  = '-.';             % bound linestyle
ls_tig  = '--';             % tightened linestyle

tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
axu = gobjects(1,2);

for i = 1:2
    axu(i) = nexttile; hold on;

    % Actual input
    pU = plot(t_u, u_act_cl(:,i), 'LineWidth', 2.2, 'Color', clr_u);

    % Symmetric bounds
    ub  =  abs(u_lim(i));
    lb  = -abs(u_lim(i));
    ubt =  abs(u_ub_tigh(i));
    lbt = -abs(u_ub_tigh(i));

    % Bound lines
    pB1 = yline(ub,  ls_bnd, 'LineWidth', 1.6, 'Color', clr_bnd);
    pB2 = yline(lb,  ls_bnd, 'LineWidth', 1.6, 'Color', clr_bnd);

    % Tightened bound lines
    pT1 = yline(ubt, ls_tig, 'LineWidth', 1.6, 'Color', clr_tig);
    pT2 = yline(lbt, ls_tig, 'LineWidth', 1.6, 'Color', clr_tig);

    ylabel(inp_names{i}, 'FontSize', 16);

    if i < 2
        %axu(i).XAxis.Visible = 'off';
    else
        xlabel('$t$ [s]', 'FontSize', 16);
    end

    set(axu(i), 'FontName','Times New Roman', 'FontSize', 14);

    % Auto y-limits with padding
    ydata_all = [u_act_cl(i,:), ub, lb, ubt, lbt];
    ylo = min(ydata_all); yhi = max(ydata_all);
    pad = 0.06 * max(1e-12, yhi - ylo);
    ylim([ylo - pad, yhi + pad]);

    % Force box ON and grid OFF for each subplot
    box(axu(i),'on');
    grid(axu(i),'off');

    % Legend only on top tile
    if i == 1
        lg = legend([pU pB1 pT1], {'Input', 'Bounds', 'Tightened'}, ...
                    'Location','best', 'FontSize', 12);
        %lg.Box = 'off';
    end
end

linkaxes(axu,'x');
%sgtitle('Inputs with Original and Tightened Bounds','FontSize',18,'FontWeight','normal');

% exportgraphics(figU,'auv_inputs_vs_bounds.pdf','ContentType','vector');
% exportgraphics(figU,'auv_inputs_vs_bounds.png','Resolution',600);
