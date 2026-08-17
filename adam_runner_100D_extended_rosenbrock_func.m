% Main script for running the ADAM optimizer

% Clear workspace and command window
clear; clc; close all;

% Create problem dimension
D = 100;

% Hyperparameters
w0 = repmat([-1.5; 2.0], D/2, 1); % Initial guess vector (100 x 1)
max_iter= 25000;  % Maximum iterations    
alpha = 0.005;   % Stepsize (learning rate)     
beta1 = 0.9;    % Exponential Decay rate 1st moment      
beta2 = 0.999;  % Exponential Decay rate 2nd moment      
epsilon = 1e-8; % Smoothing term to prevent a division by zero
tolerance = 1e-5; % Tolerance threshold. if the length (norm) of the gr<

% Sample implementation with the Rosenbrock function. Global minima: (1, 1)
f = @(x) extended_rosenbrock_f(x);
g = @(x) extended_rosenbrock_g(x);

@(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;

% Call Adam implementation
fprintf('Running ADAM optimizer... ');
tic;
[w_opt, history] = adam_optimizer(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance);
elapsed = toc;
fprintf('  Time elapsed:  %.2f seconds\n', elapsed);

% Theoretical global minimum
w_global_min = ones(D, 1);

% Print the results
% Print summary results
fprintf('Final Objective Value f(w_opt):          %.6e\n', f(w_opt));
fprintf('Final L2 Error norm(w_opt - w_global):   %.6e\n', norm(w_opt - w_global_min));
fprintf('Gradient Norm at Convergence:            %.6e\n', norm(g(w_opt)));
fprintf('Coordinate Value Range:                  [min = %.4f, max = %.4f]\n', min(w_opt), max(w_opt));

% Visualize the optimization results for high dimension
visualize_100D(f, g, history, w0, w_opt);

% Function handles for 100D Rosenbrock
function f_val = extended_rosenbrock_f(x)
    x_curr = x(1:end-1);
    x_next = x(2:end);
    f_val = sum(100 * (x_next - x_curr.^2).^2 + (1 - x_curr).^2);
end

% Function computes gradient for 100D Rosenbrock
function grad_val = extended_rosenbrock_g(x)
    D_len = length(x);
    grad_val = zeros(D_len, 1);
    
    % First element
    grad_val(1) = -400 * x(1) * (x(2) - x(1)^2) - 2 * (1 - x(1));
    
    % Middle elements
    idx = 2:D_len-1;
    grad_val(idx) = 200 * (x(idx) - x(idx-1).^2) ...
                    - 400 * x(idx) .* (x(idx+1) - x(idx).^2) ...
                    - 2 * (1 - x(idx));
                
    % Last element
    grad_val(D_len) = 200 * (x(D_len) - x(D_len-1).^2);
end