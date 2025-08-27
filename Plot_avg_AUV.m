%% plot_rsgp_20runs_script.m
% Colorful plots; average lines = blue/black; legend only on bottom subplot.

clear; clc; close all;

%% SETTINGS
Ts = 0.05;                       % sample time
dataDir = fullfile('workspace'); % folder with saved .mat files
patterns = {'Explor_RSGP_result_*.mat','Explor_RSGP_result_20.mat'};

% Base palettes
stateColors = [ ...
     0.4940 0.1840 0.5560;  % blue-ish for shadows (z)
     0.4940 0.1840 0.5560;  % orange  (theta)
     0.4940 0.1840 0.5560;  % yellow  (w)
     0.4940 0.1840 0.5560]; % purple  (q)
inputColors = [ ...
    0.4940 0.1840 0.5560;  % green  (delta_v)
    0.4940 0.1840 0.5560]; % cyan   (delta_m)

% Average colors (choose from blue/red/black) — use blue & black to avoid ref clash
avgColorsStates = [ ...
    0 0 0;   % blue
    0 0 0;   % black
    0 0 0;   % blue
    0 0 0];  % black
avgColorsInputs = [ ...
    0 0 0;   % blue
    0 0 0];  % black

refColor = [0.6350 0.0780 0.1840]; % dark red for reference

shadowFac = 0.75;  % 0..1 (lighter towards 1)
lwShadow  = 0.8; 
lwAvg     = 2.2; 
lwRef     = 1.8;

% Labels
state_names = {'$z_k$ [m]', '$\theta_k$ [rad]', '$w_k$ [m/s]', '$q_k$ [rad/s]'};
inp_names   = {'$\delta_{v,k}$', '$\delta_{m,k}$'};

% helper to lighten colors inline
lighten = @(c,f) c*(1-f) + f*[1 1 1];

%% COLLECT FILES
files = [];
for p = 1:numel(patterns)
    files = [files; dir(fullfile(dataDir, patterns{p}))]; %#ok<AGROW>
end
[~, idxUnique] = unique({files.name}, 'stable');
files = files(idxUnique);
if isempty(files), error('No result files found in "%s".', dataDir); end

%% LOAD (only xx, u_act_cl/u_cl, t, x_ref_r, average_mpc_time)
Xcell = {};  Ucell = {};  TcellX = {};  TcellU = {};
Rcell = {};  mpcTimes = [];  % collect per-run average MPC time

for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name));
    fns = fieldnames(S);
    ix = find(startsWith(fns,'ExRSGP'),1,'first');
    if isempty(ix), warning('Skipping "%s": no ExRSGP* struct.', files(k).name); continue; end
    E = S.(fns{ix});

    if isfield(E,'average_mpc_time') && ~isempty(E.average_mpc_time)
        mpcTimes(end+1) = double(E.average_mpc_time); %#ok<AGROW>
    end

    % time
    t = [];
    if isfield(E,'t') && ~isempty(E.t)
        t = E.t; if isscalar(t), t = (0:Ts:t).'; else, t = t(:); end
    end

    % states
    if ~isfield(E,'xx') || isempty(E.xx)
        warning('Skipping "%s": missing xx.', files(k).name); continue;
    end
    xx = E.xx;
    if isempty(t)
        if size(xx,1) > size(xx,2), xx = xx.'; end
        L = size(xx,2); t = (0:L-1)' * Ts;
    else
        if size(xx,2)==numel(t)
        elseif size(xx,1)==numel(t)
            xx = xx.';
        else
            if size(xx,1) > size(xx,2), xx = xx.'; end
        end
        L = min(numel(t), size(xx,2)); t = t(1:L); xx = xx(:,1:L);
    end

    % inputs (prefer u_act_cl)
    u = [];
    if isfield(E,'u_act_cl') && ~isempty(E.u_act_cl)
        u = E.u_act_cl;
    elseif isfield(E,'u_cl') && ~isempty(E.u_cl)
        u = E.u_cl;
    end
    if ~isempty(u)
        if size(u,2)==numel(t)
        elseif size(u,1)==numel(t)
            u = u.';
        else
            if size(u,1) > size(u,2), u = u.'; end
        end
        Lu = min(numel(t), size(u,2)); u = u(:,1:Lu); tU = t(1:Lu);
    else
        tU = [];
    end

    % reference x_ref_r (optional)
    r = [];
    if isfield(E,'x_ref_r') && ~isempty(E.x_ref_r)
        r = E.x_ref_r;
        if size(r,2)==numel(t)
        elseif size(r,1)==numel(t)
            r = r.';
        else
            if size(r,1) > size(r,2), r = r.'; end
        end
        r = r(:,1:min(size(r,2), numel(t)));
    end

    % store
    Xcell{end+1} = xx;     TcellX{end+1} = t;
    Ucell{end+1} = u;      TcellU{end+1} = tU;
    Rcell{end+1} = r;
end
if isempty(Xcell), error('No usable runs loaded (xx missing).'); end

%% STACK & AVERAGE — STATES (first 4)
Lx    = min(cellfun(@(X) size(X,2), Xcell));
tx    = TcellX{1}(1:Lx);
nRuns = numel(Xcell);
n_x   = size(Xcell{1},1);
m_x   = min(4, n_x);

X = zeros(m_x, Lx, nRuns);
for rID = 1:nRuns, X(:,:,rID) = Xcell{rID}(1:m_x,1:Lx); end
Xavg = mean(X,3);

% Reference (use first available)
haveRef = any(~cellfun(@isempty, Rcell));
if haveRef
    firstRef = find(~cellfun(@isempty, Rcell),1,'first');
    Rfull    = Rcell{firstRef};
    Lr       = min([size(Rfull,2), Lx]);
    Rplot    = Rfull(1:m_x, 1:Lr);
    tr       = tx(1:Lr);
end

%% STACK & AVERAGE — INPUTS (first 2 if present)
haveU = ~all(cellfun(@isempty, Ucell));
if haveU
    nonEmptyU = find(~cellfun(@isempty, Ucell), 1, 'first');
    n_u_full  = size(Ucell{nonEmptyU},1);
    m_u       = min(2, n_u_full);
    Lu        = min(cellfun(@(U) size(U,2), Ucell(~cellfun(@isempty,Ucell))));
    tu        = TcellU{nonEmptyU}(1:Lu);
    U = nan(m_u, Lu, nRuns);
    for rID = 1:nRuns
        if ~isempty(Ucell{rID})
            U(:,:,rID) = Ucell{rID}(1:m_u, 1:Lu);
        end
    end
    Uavg = mean(U,3,'omitnan');
end

%% PLOT — STATES (legend only bottom, horizontal)
fig1 = figure('Color','w','Name','States (xx)');
tiledlayout(m_x,1,'TileSpacing','compact','Padding','compact');

for i = 1:m_x
    ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    shadowC = lighten(stateColors(i,:), shadowFac);
    avgC    = avgColorsStates(i,:);

    % shadow runs
    hShadow = plot(ax, tx, squeeze(X(i,:,1)), 'Color', shadowC, 'LineWidth', lwShadow);
    for rID = 2:nRuns
        plot(ax, tx, squeeze(X(i,:,rID)), 'Color', shadowC, 'LineWidth', lwShadow);
    end

    % average (blue/black)
    hAvg = plot(ax, tx, Xavg(i,:), 'Color', avgC, 'LineWidth', lwAvg);

    % reference (if available)
    if haveRef
        hRef = plot(ax, tr, Rplot(i,:), '--', 'Color', refColor, 'LineWidth', lwRef);
    end

    % labels & fonts
    ylabel(ax, state_names{i}, 'Interpreter','latex','FontSize',20);
    if i==m_x
        xlabel(ax, 'Time (s)', 'Interpreter','latex','FontSize',20);
        if haveRef
            lg = legend(ax, [hShadow hAvg hRef], {'runs','average','reference'}, ...
                'Interpreter','latex','FontSize',20, ...
                'Location','southoutside','Orientation','horizontal');
            lg.NumColumns = 3;
        else
            lg = legend(ax, [hShadow hAvg], {'runs','average'}, ...
                'Interpreter','latex','FontSize',20, ...
                'Location','southoutside','Orientation','horizontal');
            lg.NumColumns = 2;
        end
    end
    set(ax,'TickLabelInterpreter','latex','FontSize',15);
end
linkaxes(findall(fig1,'Type','axes'),'x');

%% PLOT — INPUTS (legend only bottom, horizontal)
if haveU
    fig2 = figure('Color','w','Name','Inputs (u)');
    tiledlayout(m_u,1,'TileSpacing','compact','Padding','compact');
    for j = 1:m_u
        ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        shadowC = lighten(inputColors(j,:), shadowFac);
        avgC    = avgColorsInputs(j,:);

        % shadow runs
        hShadowU = plot(ax, tu, squeeze(U(j,:,1)), 'Color', shadowC, 'LineWidth', lwShadow);
        for rID = 2:nRuns
            if ~all(isnan(U(j,:,rID)))
                plot(ax, tu, squeeze(U(j,:,rID)), 'Color', shadowC, 'LineWidth', lwShadow);
            end
        end

        % average (blue/black)
        hAvgU = plot(ax, tu, Uavg(j,:), 'Color', avgC, 'LineWidth', lwAvg);

        ylabel(ax, inp_names{j}, 'Interpreter','latex','FontSize',20);
        if j==m_u
            xlabel(ax, 'Time (s)', 'Interpreter','latex','FontSize',20);
            lg2 = legend(ax, [hShadowU hAvgU], {'runs','average'}, ...
                'Interpreter','latex','FontSize',20, ...
                'Location','southoutside','Orientation','horizontal');
            lg2.NumColumns = 2;
        end
        set(ax,'TickLabelInterpreter','latex','FontSize',15);
    end
    linkaxes(findall(fig2,'Type','axes'),'x');
end

%% REPORT AVERAGE MPC TIME
if ~isempty(mpcTimes)
    avgMpc = mean(mpcTimes);
    fprintf('Average MPC time over %d runs: %.6f s\n', numel(mpcTimes), avgMpc);
else
    warning('average_mpc_time not found in the loaded structs.');
end
