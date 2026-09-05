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
lambda = 0.1; % The Decoupled weight decay (for Adam W)

% Sample implementation with the Rosenbrock function. Global minima: (1, 1)
f = @(x) extended_rosenbrock_f(x);
g = @(x) extended_rosenbrock_g(x);

@(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;

% Call Adam implementation
fprintf('Running the benchmarker.... \n ');
w_global_min = ones(D, 1);

% Registry of optimizers to benchmark.
optimizers = {
    struct('name', 'Adam',                      'fn', @(g, w0) adam_optimizer(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance))
    struct('name', 'AdamW',                      'fn', @(g, w0) adam_optimizer_W(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance, lambda))
    struct('name', 'AMSGrad',                    'fn', @(g, w0) adam_optimizer_AMSGrad(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance))
    struct('name', 'ADAM with Double Gradient',  'fn', @(g, w0) adam_optimizer_DG(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance))
};

%loop through all the optimizers
for k = 1:numel(optimizers)
    opt = optimizers{k};
    fprintf('========Results %s=======\n', opt.name);
    tic;
    [w_opt, history] = opt.fn(g, w0);
    benchmark_values(f, g, w_global_min, w_opt, history);
end



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

function benchmark_values(f, g, w_global_min, w, history)

    elapsed = toc;
    fprintf('Time elapsed:  %.2f seconds\n', elapsed);

    % Print summary results
    fprintf('Final Objective Value f(w_opt):          %.6e\n', f(w));
    fprintf('Final L2 Error norm(w_opt - w_global):   %.6e\n', norm(w - w_global_min));
    fprintf('Gradient Norm at Convergence:            %.6e\n', norm(g(w)));
    fprintf('Coordinate Value Range:                  [min = %.4f, max = %.4f]\n\n', min(w), max(w));

    % Visualize the optimization results for high dimension
    %visualize_100D(f, g, history, w0, w_opt);
end
