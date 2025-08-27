function noise = truncated_gaussian_noise(sz, variance, bound_factor)
    % Generate zero-mean truncated Gaussian noise with specified variance
    % sz: output size
    % variance: target variance before truncation
    % bound_factor: bounds are at +/- bound_factor * std
    %               (e.g., bound_factor=2 means [-2σ, 2σ])
    stddev = sqrt(variance);
    a = -bound_factor * stddev;
    b =  bound_factor * stddev;
    noise = stddev * randn(sz);

    % Truncate values outside [a, b]
    noise(noise < a) = a;
    noise(noise > b) = b;
    
    % Note: after truncation, variance will be slightly less than specified
end
