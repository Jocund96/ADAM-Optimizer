% ADAM optimizer by Bauhaus University

% INPUTS 
% grad_f    : Gradient function handle
% state : optimizer memory, empty [] on the first call
% hp    : alpha, beta1, beta2, epsilon

function [params, state] = adam_optimizer_step(params, grads, state, t, hp)

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
    
    %update the parameters
    params.(key) = params.(key) - hp.learning_rate * m_hat ./ (sqrt(v_hat) + hp.epsilon);

end
% Loop end

end
%end function