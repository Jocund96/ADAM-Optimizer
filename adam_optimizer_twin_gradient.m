% ADAM optimizer by Bauhaus University

% INPUTS 
% grad_f    : Gradient function handle
% w01, w02  : Initial guess vectors (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr< 

function [w_opt, history, verbose] = adam_optimizer_twin_gradient(grad_f, w01, w02, max_iter, alpha, beta1, beta2, epsilon, tolerance)
% Define validation rules and DEFAULT values
arguments
        grad_f   function_handle
        w01      double
        w02      double
        max_iter double = 1500       
        alpha    double = 0.001      
        beta1    double = 0.9        
        beta2    double = 0.999      
        epsilon  double = 1e-8   
        tolerance double = 1e-6
end

% Initialize variables for the optimizer
w1 = w01(:); % Current weight
w2 = w02(:); % Current weight
m1 = zeros(size(w01)); % First moment vector  m_0
m2 = zeros(size(w02)); % First moment vector  m_0
v1 = zeros(size(w01)); % Second moment vector  v_0
v2 = zeros(size(w02)); % Second moment vector  v_0
t = 0; %loop iterations 

% Initialize our custom history ledger folder
history.w1 = [];
history.w2 = [];
verbose.iteration = 0;

% Gradient at the initial guess; refreshed at the end of every
% iteration so the while-condition and the loop body share one
% evaluation per point instead of computing grad_f(w) twice.
gradient1 = grad_f(w1);
gradient2 = grad_f(w2);

% OPTIMIZATION LOOP
% Loop start : terminate at convergence
while (norm(gradient1) > tolerance) && (norm(gradient2) > tolerance) && (t < max_iter)
    % Timestep counter
    t = t+1;
    
    % Update exponential moving averages of the gradient
    m1 = beta1*m1+(1-beta1)*gradient1; %first moment estimate
    m2 = beta1*m2+(1-beta1)*gradient2; %first moment estimate

    % Update squared gradient
    v1 = beta2*v1+(1-beta2)*(gradient1.^2); %second moment estimate
    v2 = beta2*v2+(1-beta2)*(gradient2.^2); %second moment estimate

    % Compute bias-corrected first moment estimate
    m_hat1 = m1/(1-beta1^t);
    m_hat2 = m2/(1-beta1^t);

    % Compute bias-corrected second moment estimate
    v_hat1 = v1/(1-beta2^t);
    v_hat2 = v2/(1-beta2^t);   
    
    % Unnormalized Adam step vectors (with learning rate alpha included)
    s1 = -alpha * m_hat1 ./ (sqrt(v_hat1) + epsilon);
    s2 = -alpha * m_hat2 ./ (sqrt(v_hat2) + epsilon);
    
    s1_sq = dot(s1, s1);
    s2_sq = dot(s2, s2);
    if (s1_sq < epsilon^2) || (s2_sq < epsilon^2)
        break;
    end
    
    % Displacement and inner products for 2x2 distance minimization
    delta = w1 - w2;
    s1_s2 = dot(s1, s2);
    s1_d  = dot(s1, delta);
    s2_d  = dot(s2, delta);
    
    % Normalized determinant: 1 - cos^2(theta)
    det_sys = s1_sq * s2_sq - s1_s2^2;
    normalized_det = det_sys / (s1_sq * s2_sq);
    
    if normalized_det > 1e-6
        % Solve for multipliers
        gamma1 = (-s1_d * s2_sq + s1_s2 * s2_d) / det_sys;
        gamma2 = ( s2_d * s1_sq - s1_s2 * s1_d) / det_sys;
        
        % Bound multipliers near 1.0 so each trajectory always moves forward
        gamma1 = max(0.5, min(gamma1, 3.5));
        gamma2 = max(0.5, min(gamma2, 3.5));
    else
        gamma1 = 1.0;
        gamma2 = 1.0;
    end
    
    % Update parameters
    w1 = w1 + gamma1 * s1;
    w2 = w2 + gamma2 * s2;

    % Record the history
    history.w1(:,t) = w1;
    history.w2(:,t) = w2;

    % Gradient at the updated parameters: used by the next
    % while-condition check and the next iteration body
    gradient1 = grad_f(w1);
    gradient2 = grad_f(w2);

end
% Loop end


% Return the smallest gradient norm
if norm(gradient1) <= norm(gradient2)
    w_opt = w1;
else
    w_opt = w2;
end

verbose.iteration = t;
end

