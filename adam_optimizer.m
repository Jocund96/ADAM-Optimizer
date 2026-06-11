%----------ADAM optimizer by Bauhaus University

% --- INPUTS ---
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% --- Convergence Criteria ---
% max_iter  : Maximum number of iterations
%           OR
% error   : Acceptable error in the Objective function.

function [w_opt, history] = adam_optimizer(grad_f, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance)
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

%-----OPTIMIZATION LOOP

% Loop start : terminate at convergence
while (norm(grad_f(w)) > tolerance) && (t < max_iter)

t = t+1;
%calculate the gradient from the given grad_f
gradient = grad_f(w);

%update the biased moment estimates
m = beta1*m+(1-beta1)*gradient; %first moment estimate
v = beta2*v+(1-beta2)*(gradient.^2); %second moment estimate

%compute the bias corrected moments
m_hat = m/(1-beta1^t);
v_hat = v/(1-beta2^t);

%update the parameters
w = w-alpha*m_hat./(sqrt(v_hat) + epsilon);

%record the history
history.w(:,t) = w;

end
% Loop end

w_opt = w;  %return the optimized values
fprintf('Number of iterations to convergence: (%.4d)\n', t);
end
%end function