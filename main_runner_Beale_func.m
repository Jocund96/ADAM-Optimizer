%  Main script for running the ADAM optimizer

% Clear workspace and command window
clear; clc;

% Define function and gradient
% Sample implementation with the Beale function. Global minima: (3, 0.5)
f = @(x) (1.5 - x(1) + x(1)*x(2))^2 + (2.25 - x(1) + x(1)*x(2)^2)^2 + (2.625 - x(1) + x(1)*x(2)^3)^2;
g = @(x) [2*x(1)*x(2)^6+2*x(1)*x(2)^4-4*x(1)*x(2)^3-2*x(1)*x(2)^2-4*x(1)*x(2)+6*x(1)+21/4*x(2)^3+9/2*x(2)^2+3*x(2)-51/4; 
    6*x(1)^2*x(2)^5+4*x(1)^2*x(2)^3-6*x(1)^2*x(2)^2-2*x(1)^2*x(2)-2*x(1)^2+63/4*x(1)*x(2)^2+9*x(1)*x(2)+3*x(1)];

f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
g = @(x) [ -2*(1-x(1)) - 400*x(1)*(x(2) - x(1)^2);
    200*(x(2) - x(1)^2) ];

% Set loop parameters
w0 = [1.5; 0.0];    % Initial guess
w0 = [-1.5; 2.0];    % Initial guess
w = w0;             % Current weight vector
t = 0;              % Iteration counter
max_iter = 25000;    % Maximum number of iterations
epsilon_t = 1e-5;   % Convergence tolerance threshold

% Initial gradient evaluation
gt = g(w);

% Initialize vectors to store intermediate steps needed for visualization
w_all = [w];
f_all = [f(w)];

% Call Adam implementation
adam_solver_JO

% Print the results
fprintf('Optimization Completed!\n');
fprintf('Final Coordinates: (%.4f, %.4f)\n', w(1), w(2));
fprintf('Global Minimum is at (3.0000, 0.5000)\n');
fprintf('Number of iterations to convergence: %d', t)

% Call Visualization script
visualize_adam_JO