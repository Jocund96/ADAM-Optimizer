% ADAM optimizer by Bauhaus University

% INPUTS 
% f         : Objective function handle
% grad_f    : Gradient function handle
% w0        : Initial guess (vector of size N x 1)

% Convergence Criteria
% max_iter  : Maximum number of iterations
% OR/AND
% epsilon_t : Tolerance threshold. if the length (norm) of the gr< 

function [params, state] = adam_optimizer_AMSGrad_step(params, grads, state, t, hp)

fields = fieldnames(params);

%% Initialisierung ADAM Variablen
if isempty(state)
    for i = 1:length(fields)
        state.m.(fields{i}) = zeros(size(params.(fields{i})));
        state.v.(fields{i}) = zeros(size(params.(fields{i})));
        %For the AMSGrad upgrade
        state.vmax.(fields{i}) = zeros(size(params.(fields{i})));
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
    
    state.vmax.(key) = max(state.vmax.(key), state.v.(key));
    m_hat = state.m.(key)/(1-hp.beta1^t);
    v_hat = state.vmax.(key)/(1-hp.beta2^t);

    %w = w - alpha * m_hat ./ (sqrt(v_hat_max) + epsilon);  % use max
    params.(key) = params.(key) - hp.learning_rate * m_hat ./ (sqrt(v_hat) + hp.epsilon);

end
% Loop end
end
%end function