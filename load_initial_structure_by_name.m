function initdata = load_initial_structure_by_name(filename)
% Loads a specific initialization structure from the "initial_Structure" folder.
% Usage:
%   initdata = load_initial_structure_by_name('IS_X_Y_Z.mat');
%
% Input:
%   filename - Name of the .mat file to load (e.g., 'IS_2p45e-05_2025-07-21_154312.mat')
%
% Output:
%   initdata - Loaded structure

    % --- Folder to load from ---
    loadfolder = 'initial_Structure';
    filepath = fullfile(loadfolder, filename);

    if ~exist(filepath, 'file')
        error('File "%s" does not exist in "%s".', filename, loadfolder);
    end

    S = load(filepath);
    if isfield(S, 'initdata')
        initdata = S.initdata;
        fprintf('Loaded initialization structure from %s\n', filepath);
    else
        error('File "%s" does not contain "initdata" structure.', filename);
    end
end
