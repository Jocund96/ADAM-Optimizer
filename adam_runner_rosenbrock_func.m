% Main script for running the ADAM optimizer

% Clear workspace and command window
clear; clc; close all;

% Create function handles pointing to your separate files

% Hyperparameters

w0 = [-1.5; 2.0];
max_iter= 15000;       
alpha = 0.01;      
beta1 = 0.9;        
beta2 = 0.999;      
epsilon = 1e-8;
tolerance = 1e-5;

% Sample implementation with the Rosenbrock function. Global minima: (1, 1)
f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
g = @(x) [ -2*(1-x(1)) - 400*x(1)*(x(2) - x(1)^2);
              200*(x(2) - x(1)^2) ];

% Call your Adam implementation
[w_opt, history] = adam_optimizer(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance);

% Theoretical global minimum
w_global_min = fminsearch(f, w0);

% Print the results
fprintf('Theoretical global minimum found by fminsearch: (%.4f, %.4f)\n', w_global_min(1), w_global_min(2));
fprintf('Final Coordinates found by Adam optimizer:    (%.4f, %.4f)\n', w_opt(1), w_opt(2));

% Instead of calling visualize_adam() for plotting
visualize(f,history,w0,w_opt);