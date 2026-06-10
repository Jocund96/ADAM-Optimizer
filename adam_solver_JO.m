% ADAM optimizer by Bauhaus University

% Input variables
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr< 

% Initialization
alpha = 0.01; % Stepsize
beta1 = 0.9;  % Exponential Decay rate 1st moment
beta2 = 0.999; % Exponential Decay rate 2nd moment
lambda = 1-1e-8; % Exponential Decay rate of beta1
epsilon_s = 1e-8; % Smoothing term to prevent a division by zero

% Initialize variables for the optimizer
m = zeros(size(w)); % First moment vector  m_0
v = zeros(size(w)); % Second moment vector  v_0

%% Optimization loop
% 1. Loop start : terminate at convergence
while (norm(gt)>epsilon_t) && (t < max_iter)
    
    % Timestep counter
    t = t + 1;

    % 2. Decay the first moment running average coefficient
    beta1_t = beta1 * lambda^(t-1);

    % 3. Get gradients w.r.t. stochastic objective at timestep t
    gt = g(w);

    % 4. Update exponential moving averages of the gradient 
    m = beta1_t * m + (1-beta1_t) * gt;

    % 5. Update squared gradient 
    v = beta2 * v + (1-beta2) * (gt.^2);
    
    % 6. Compute bias-corrected first moment estimate
    m_est = m/(1-beta1^t);

    % 7. Compute bias-corrected second moment estimate
    v_est = v/(1-beta2^t);

    % 8. Update parameters
    w = w - alpha * m_est./(sqrt(v_est)+epsilon_s);
    
    % Store new points for visualization
    w_all = [w_all, w];
    f_all = [f_all, f(w)];
    
    % Evaluate gradient for the next loop condition check
    gt = g(w);
end
% Loop end