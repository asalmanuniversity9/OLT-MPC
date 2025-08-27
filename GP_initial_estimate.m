function [gprMdl,XR,YR] = GP_initial_estimate(B,g,plot_index,plot_bar,N,sigma_noise)

if nargin < 3                 % nargin = # of inputs actually passed
    plot_index = false;                   % initialize a default
    plot_bar = false;
    N = 50;
elseif nargin < 4
    plot_bar = false;
    N = 50;
elseif nargin < 5
    N = 50;
end
%plot_bar = false;
%% ---------- physical / operating envelopes ----------
z_min  =   0;    z_max  =  50;     % depth   [m]
th_lim = 0.17;                     % |theta| ≤ 10° ≈ 0.17 rad
w_lim  = 1.0;                      % |w|     ≤ 1   rad s⁻¹
q_lim  = 0.5;                      % |q|     ≤ 0.5 rad s⁻¹


%N  = 50;                         % needs fewer points
dbclear if naninf
lhs = lhsdesign(N,4,'criterion','maximin');

z     =  z_min  + lhs(:,1)*(z_max-z_min);
theta = -th_lim + lhs(:,2)*2*th_lim;
w     = -w_lim  + lhs(:,3)*2*w_lim;
q     = -q_lim  + lhs(:,4)*2*q_lim;

Xtrain = [z theta w q];
XR = Xtrain;
%noise_t =sigma_noise*[randn(1,length(Xtrain)); randn(1,length(Xtrain))];
noise_t = truncated_gaussian_noise([2,length(Xtrain)], sigma_noise, 3*sigma_noise);
Ytrain = zeros(N,2);
YR = zeros(N,2);
for n = 1:N
    Ytrain(n,:) = (pinv(B)*g(Xtrain(n,:)'))+noise_t(:,n);     % the nonlinear term
end

gprMdl = trainGPRmulti(Xtrain, Ytrain, ...
    'Basis','constant','FitMethod','sd',...
    'PredictMethod','sd','KernelFunction','squaredexponential','Standardize',1);
for p = 1:2
    YR(:,p) = predict(gprMdl{p}, Xtrain);                 % mean only
end
%% === 0.  prerequisites ===
% – assume g(), the limits, and gprMdl (cell array returned by trainGPRmulti)
%   already exist in the workspace.

%% 1.  build a test set inside the same box
Ntest  = 1500;                                              % # test points
lhsT   = lhsdesign(Ntest,4,'criterion','maximin');

zT     = z_min  + lhsT(:,1)*(z_max - z_min);
thetaT = -th_lim + lhsT(:,2)*2*th_lim;
wT     = -w_lim  + lhsT(:,3)*2*w_lim;
qT     = -q_lim  + lhsT(:,4)*2*q_lim;

Xtest  = [zT thetaT wT qT];                                 % [Ntest×4]

%% 2.  evaluate the truth g(x)  and the GP predictions
Ytrue  = zeros(Ntest,2);
for i = 1:Ntest
    Ytrue(i,:) = pinv(B)*g(Xtest(i,:)');
end

Ypred  = zeros(Ntest,2);
for p = 1:2
    [Ypred(:,p),sd(:,p)] = predict(gprMdl{p}, Xtest);                 % mean only
    %[mu,sd] = predict(gprMdl{p}, Xtest);    % mu, sd are Ntest×1
    varPred(:,p) = sd(:,p).^2;
end

if(plot_index)
    %% 3.  scatter (truth vs GP) for each output
    figure('Name','GP vs Truth — all four outputs','Position',[100 100 900 700])
    titles = {'g₁(x)','g₂(x)','g₃(x)','g₄(x)'};
    
    for p = 1:2
        subplot(2,1,p)
        scatter(Ytrue(:,p), Ypred(:,p), 14,varPred(:,p),'filled'); hold on
        if(plot_bar)
            errorbar(Ytrue(:,p),Ypred(:,p), 2*sd(:,p), '.', 'CapSize',0)
        end
        xy = [min(Ytrue(:,p)) max(Ytrue(:,p))];
        plot(xy,xy,'k--','LineWidth',1);                        % 45° reference
        xlabel('Truth  g_i(x)');  ylabel('GP prediction \mu_i');
        title(titles{p});
        grid on
    end
    sgtitle('GP mean prediction vs ground-truth nonlinear term')
end
end