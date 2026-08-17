% Main script for running the ADAM optimizer

% Clear workspace and command window
clear; clc; close all;

% Create function handles pointing to your separate files

%% ── Hyperparameters 

w0 = 20.0;
max_iter= 1500;       
alpha = 0.5;      
beta1 = 0.9;        
beta2 = 0.999;      
epsilon = 1e-8;
tolerance = 1e-6;


f = @(x) x^2 + 4*x + 4;
g = @(x) 2*x + 4;

% Call your Adam implementation
[w_opt, history] = adam_optimizer(g, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance);
% Print the results
w_opt
history

% Instead of calling visualizer() for a 1D function ---
figure;
% Plot the tracking of the weight over time
plot(history.w, 'b.-', 'LineWidth', 1.5);
title('ADAM Parameter Tracking for 1D Quadratic');
xlabel('Iteration (t)'); ylabel('Parameter Value (w)');
grid on;