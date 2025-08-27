function save_initial_structure()
% Save initialization variables for MPC/AUV simulation in a structured .mat file
% Format: initial_Structure/IS_X_Y_Z.mat
% X: Cq_final (tube constant), Y: date, Z: time

    % --- List variables to include ---
    vars = {'A','B','Q','R','z_min','z_max','th_lim','w_lim','q_lim','g','Gamma',...
            'C1','C2','Cq_final','gprMdl','ytrain','xtrain','KernelInformation','f','sigmaF0','Sigma_t','S_t','G_t','sigma_t_n','ytrain_0'};
    initdata = struct();
    
    % --- Collect variables from base workspace ---
    for k = 1:numel(vars)
        try
            initdata.(vars{k}) = evalin('base', vars{k});
        catch
            % If not found, just skip (except for Cq_final which is mandatory)
            if strcmp(vars{k},'Cq_final')
                error('Cq_final must be defined in the base workspace.');
            end
        end
    end

    % --- Folder creation ---
    savefolder = 'initial_Structure';
    if ~exist(savefolder,'dir')
        mkdir(savefolder);
    end

    % --- Filename construction ---
    X = initdata.Cq_final;
    Xstr = sprintf('%.2e', X);       % Scientific notation
    Xstr = strrep(Xstr, '.', 'p');   % safer for filenames
    Xstr = strrep(Xstr, '-', 'm');   % for negatives
    
    dt  = datetime('now');
    Y   = datestr(dt, 'yyyy-mm-dd');
    Z   = datestr(dt, 'HHMMSS');
    fname = sprintf('IS_%s_%s_%s.mat', Xstr, Y, Z);

    % --- Save ---
    save(fullfile(savefolder, fname), 'initdata');
    fprintf('Saved initialization structure as %s\n', fullfile(savefolder, fname));
end
