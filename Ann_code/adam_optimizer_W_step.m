% ADAM optimizer by Bauhaus University

% INPUTS 
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr< 

%% needs a lambda in the hp

function [params, state] = adam_optimizer_W_step(params, grads, state, t, hp)

lambda = 0; %placeholder  change later

fields = fieldnames(params);

%% Initialisierung ADAM Variablen
if isempty(state)
    for i = 1:length(fields)
        state.m.(fields{i}) = zeros(size(params.(fields{i})));
        state.v.(fields{i}) = zeros(size(params.(fields{i})));
    end
end

% OPTIMIZATION Step
%% Loop over the weight and bias fields (W1, b1, W2, ...)
for i = 1: length(fields)

    key = fields{i};

    g = grads.(key);

    % Update exponential moving averages of the gradient
    state.m.(key) = hp.beta1*state.m.(key)+(1-hp.beta1)*g; %first moment estimate

    % Update squared gradient
    state.v.(key) = hp.beta2*state.v.(key)+(1-hp.beta2)*(g.^2); %second moment estimate
    
    % Compute bias-corrected first moment estimate
    m_hat = state.m.(key)/(1-hp.beta1^t);

    % Compute bias-corrected second moment estimate
    v_hat = state.v.(key)/(1-hp.beta2^t);
    
    %update the parameters with Decoupled Weight Decay (AdamW)
    %w = w - alpha * m_hat ./ (sqrt(v_hat) + epsilon) -             alpha * lambda * w;
    params.(key) = params.(key) - hp.learning_rate * m_hat ./ (sqrt(v_hat) + hp.epsilon) - hp.learning_rate*lambda*params.(key);


end
% Loop end

end
%end function