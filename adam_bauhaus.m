%----------ADAM optimizer by Bauhaus University

% --- INPUTS ---
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)
% max_iter  : Maximum number of iterations

% --- INITIALIZATION ---
alpha = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;