%----------ADAM optimizer by Bauhaus University

% --- INPUTS ---
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)
% max_iter  : Maximum number of iterations

% --- INITIALIZATION ---
alpha = 0.001; % Learning rate
beta1 = 0.9;  % Exponential Decay rates
beta2 = 0.999;
epsilon = 1e-8;

% Initialize variables for the optimizer
w = w0; % Current weight
m = zeros(size(w0)); % First moment vector  m_0
v = zeros(size(w0)); % Second moment vector  v_0

%-----OPTIMIZATION LOOP
