% ADAM optimizer by Bauhaus University

% INPUTS
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr<

function [w_opt, history, verbose] = adam_optimizer_preconditioned(grad_f, w0, max_iter, alpha, beta1, beta2, epsilon, tolerance)
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
H = numerical_hessian(grad_f, w);    % estimate the Hessian at current point
H_reg = regularize_hessian(H, numel(w0));

% Full preconditioning (Newton-like), guarded against an indefinite
% Hessian (common away from the minimum on non-convex problems like
% Rosenbrock, where a raw Newton step can point uphill)
precond_grad = descent_direction(H_reg, gradient);

% OPTIMIZATION LOOP
% Loop start : terminate at convergence
while (norm(precond_grad) > tolerance) && (t < max_iter)
    % Timestep counter
    t = t+1;

    % Update exponential moving averages of the gradient
    m = beta1*m+(1-beta1)*precond_grad; %first moment estimate

    % Update squared gradient
    v = beta2*v+(1-beta2)*(precond_grad.^2); %second moment estimate

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

    if mod(t, 50) == 1   % every 50 steps
        H = numerical_hessian(grad_f, w);    % estimate the Hessian at current point
        H_reg = regularize_hessian(H, numel(w0));
    end
    % Use the same (regularized) H for 50 steps, but re-check every step
    % that it still yields a descent direction against the fresh gradient
    precond_grad = descent_direction(H_reg, gradient);

end
% Loop end

w_opt = w;  %return the optimized values
%fprintf('Number of iterations to convergence: (%.4d)\n', t);
verbose.iteration = t;
end
%end function

% Numerically estimates the Hessian by central-differencing the gradient:
function H_num = numerical_hessian(grad_f, w, h)
    if nargin < 3
        h = 1e-5;
    end
    N = numel(w);
    H_num = zeros(N);
    for j = 1:N
        e_j = zeros(N, 1);
        e_j(j) = 1;
        g_plus  = grad_f(w + h * e_j);
        g_minus = grad_f(w - h * e_j);
        H_num(:, j) = (g_plus - g_minus) / (2 * h);
    end
    H_num = (H_num + H_num') / 2;
end

% Rosenbrock's Hessian (and non-convex Hessians generally) is only
% positive definite near the minimum. Away from it, Newton's step
% H \ gradient can point uphill. Damp H toward positive definiteness by
% adding a multiple of the identity (Levenberg-Marquardt style) until a
% Cholesky factorization succeeds.
function H_reg = regularize_hessian(H, N)
    mu = 0;
    max_tries = 20;
    for attempt = 1:max_tries
        H_reg = H + mu * eye(N);
        [~, p] = chol(H_reg);
        if p == 0
            return;
        end
        if mu == 0
            mu = 1e-6;
        else
            mu = mu * 10;
        end
    end
    % Damping alone could not certify positive-definiteness within the
    % iteration budget; H_reg here is dominated by mu*I, which makes the
    % preconditioned step below reduce to (approximately) a scaled
    % gradient step.
end

% Even a damped/PD H can, due to numerical error, yield a step that does
% not correlate with the true gradient. Verify gradient' * p > 0 (the
% condition for -p to be a descent direction) and fall back to the raw
% gradient otherwise.
function p = descent_direction(H_reg, gradient)
    p = H_reg \ gradient;
    if ~all(isfinite(p)) || (gradient' * p) <= 0
        p = gradient;
    end
end
