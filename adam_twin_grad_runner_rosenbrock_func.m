% Main script for running the ADAM Twing Gradient optimizer

% Clear workspace and command window
clear; clc; close all;

% Create function handles pointing to your separate files

% Hyperparameters
w01 = [-1.5; 2.0]; % Initial guess #1
w02 = [ 1.5; 2.0]; % Initial guess #2
w_ref = [1.0; 1.0]; % Theoretical minimum
max_iter= 15000;  % Maximum iterations    
alpha = 0.01;   % Stepsize     
beta1 = 0.9;    % Exponential Decay rate 1st moment      
beta2 = 0.999;  % Exponential Decay rate 2nd moment      
epsilon = 1e-8; % Smoothing term to prevent a division by zero
tolerance = 1e-5; % Tolerance threshold. if the length (norm) of the gr<

% Sample implementation with the Rosenbrock function. Global minima: (1, 1)
f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
g = @(x) [ -2*(1-x(1)) - 400*x(1)*(x(2) - x(1)^2);
              200*(x(2) - x(1)^2) ];

% Call Adam implementation
[w_opt, history, verbose] = adam_optimizer_twin_gradient(g, w01, w02, max_iter, alpha, beta1, beta2, epsilon, tolerance);

% Print the results
fprintf('Known Global Minimum:                         (%.4f, %.4f)\n', w_ref(1), w_ref(2));
fprintf('Final Coordinates found by Twin Adam:         (%.4f, %.4f)\n', w_opt(1), w_opt(2));
fprintf('Iterations to convergence:                    %d\n', verbose.iteration);

% Instead of calling visualize_adam() for plotting
visualize_twin(f, history, w01, w02, w_opt, w_ref);