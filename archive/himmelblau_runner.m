% -------- Main script for running the ADAM optimizer

% Clear workspace and command window
clear; clc; close all;

% Create function handles pointing to your separate files

%% ── Hyperparameters ─────────────────────────────────────

w0 = [-1.5; 2.0];
max_iter= 1500;       
alpha = 0.01;      
beta1 = 0.9;        
beta2 = 0.999;      
epsilon = 1e-8;
tolerance = 1e-6;


f = @(x) (x(1)^2 + x(2) - 11)^2 + (x(1) + x(2)^2 - 7)^2;
g = @(x) [ 4*x(1)*(x(1)^2 + x(2) - 11) + 2*(x(1) + x(2)^2 - 7);
             2*(x(1)^2 + x(2) - 11)   + 4*x(2)*(x(1) + x(2)^2 - 7) ];

% Call your Adam implementation
[w_opt, history] = adam_optimizer(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance);
% Print the results
w_opt
history

% --- Instead of calling visualizer() for a 1D function ---
visualizer(f,history,w0,w_opt);