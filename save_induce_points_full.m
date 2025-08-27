function save_name = save_induce_points_full(xtrain, ytrain, gprMdl, n_R, history_dir)
% Save induced points, ytrain, and gprMdl with date-time versioning.
%
% INPUTS:
%   xtrain     : Induced points (matrix)
%   ytrain     : Corresponding y values (matrix)
%   gprMdl     : Trained GP model object or struct
%   n_R        : Number of induced points (for filename)
%   history_dir: Folder to store history (default: 'induce_point_history')
%
% OUTPUTS:
%   save_name  : Full file name of the saved .mat file

    if nargin < 5 || isempty(history_dir)
        history_dir = 'induce_point_history';
    end
    if ~exist(history_dir, 'dir')
        mkdir(history_dir);
    end

    % Date-time string for unique versioning
    dt = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    dt_str = char(dt);

    % File name
    save_name = sprintf('induce_points_nR%d_%s.mat', n_R, dt_str);

    % Save all
    induce_points = xtrain; 
    save(fullfile(history_dir, save_name), ...
        'induce_points', 'ytrain', 'gprMdl', 'n_R', 'dt_str');
    fprintf('[save_induce_points_full] Saved: %s\n', fullfile(history_dir, save_name));
end
