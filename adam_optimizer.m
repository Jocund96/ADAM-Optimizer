% ADAM optimizer by Bauhaus University

% INPUTS 
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr< 

function [w_opt, history, verbose] = adam_optimizer(grad_f, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance)
% Define validation rules and DEFAULT values
arguments
        grad_f   function_handle
        w0       double
        max_iter double = 1500       
        alpha    double = 0.001      
        beta1    double = 0.9        
        beta2    double = 0.999      
        epsilon  double = 1e-8   
        tolerance double = 1e-6
end


% Initialize variables for the optimizer
w = w0; % Current weight
m = zeros(size(w0)); % First moment vector  m_0
v = zeros(size(w0)); % Second moment vector  v_0
t = 0; %loop iterations 

% Initialize our custom history ledger folder
history.w = [];
verbose.iteration = 0;

% Gradient at the initial guess; refreshed at the end of every
% iteration so the while-condition and the loop body share one
% evaluation per point instead of computing grad_f(w) twice.
gradient = grad_f(w);

% OPTIMIZATION LOOP
% Loop start : terminate at convergence
while (norm(gradient) > tolerance) && (t < max_iter)
    % Timestep counter
    t = t+1;

    % Update exponential moving averages of the gradient
    m = beta1*m+(1-beta1)*gradient; %first moment estimate

    % Update squared gradient
    v = beta2*v+(1-beta2)*(gradient.^2); %second moment estimate
    
    % Compute bias-corrected first moment estimate
    m_hat = m/(1-beta1^t);

    % Compute bias-corrected second moment estimate
    v_hat = v/(1-beta2^t);
    
    %update the parameters
    w = w - alpha * m_hat./(sqrt(v_hat) + epsilon);

    %record the history
    history.w(:,t) = w;

    % Gradient at the updated parameters: used by the next
    % while-condition check and the next iteration body
    gradient = grad_f(w);

end
% Loop end

w_opt = w;  %return the optimized values
%fprintf('Number of iterations to convergence: (%.4d)\n', t);
verbose.iteration = t;
end
%end function