function [p1, p2, state] = adam_optimizer_twin_step(p1, p2, g1, g2, state, t, hp)
% Twin ADAM: one step for two networks p1, p2 with gradients g1, g2.
% Both take an Adam step s1, s2. The step lengths are scaled by gamma1,
% gamma2 so that the two networks end as close to each other as possible:
%     min || (w1 + gamma1*s1) - (w2 + gamma2*s2) ||^2      (2x2 system)
% hp : learning_rate, beta1, beta2, epsilon

fields = fieldnames(p1);

% Moment vectors for both networks (zero on the first call)
if isempty(state)
    for i = 1:numel(fields)
        k = fields{i};
        state.m1.(k) = zeros(size(p1.(k)));  state.v1.(k) = zeros(size(p1.(k)));
        state.m2.(k) = zeros(size(p2.(k)));  state.v2.(k) = zeros(size(p2.(k)));
    end
end

% Adam step vectors s1, s2 and the inner products of the 2x2 system
s1_sq = 0;  s2_sq = 0;  s1_s2 = 0;  s1_d = 0;  s2_d = 0;
for i = 1:numel(fields)
    k = fields{i};
    state.m1.(k) = hp.beta1*state.m1.(k) + (1-hp.beta1)*g1.(k);
    state.m2.(k) = hp.beta1*state.m2.(k) + (1-hp.beta1)*g2.(k);
    state.v1.(k) = hp.beta2*state.v1.(k) + (1-hp.beta2)*g1.(k).^2;
    state.v2.(k) = hp.beta2*state.v2.(k) + (1-hp.beta2)*g2.(k).^2;

    m_hat1 = state.m1.(k)/(1-hp.beta1^t);   v_hat1 = state.v1.(k)/(1-hp.beta2^t);
    m_hat2 = state.m2.(k)/(1-hp.beta1^t);   v_hat2 = state.v2.(k)/(1-hp.beta2^t);

    s1.(k) = -hp.learning_rate * m_hat1 ./ (sqrt(v_hat1) + hp.epsilon);
    s2.(k) = -hp.learning_rate * m_hat2 ./ (sqrt(v_hat2) + hp.epsilon);

    d = p1.(k) - p2.(k);
    s1_sq = s1_sq + sum(s1.(k)(:).^2);
    s2_sq = s2_sq + sum(s2.(k)(:).^2);
    s1_s2 = s1_s2 + sum(s1.(k)(:) .* s2.(k)(:));
    s1_d  = s1_d  + sum(s1.(k)(:) .* d(:));
    s2_d  = s2_d  + sum(s2.(k)(:) .* d(:));
end

% Step multipliers, bounded so both networks always move forward
det_sys = s1_sq * s2_sq - s1_s2^2;
if s1_sq > 0 && s2_sq > 0 && det_sys / (s1_sq * s2_sq) > 1e-6
    gamma1 = max(0.5, min((-s1_d * s2_sq + s1_s2 * s2_d) / det_sys, 3.5));
    gamma2 = max(0.5, min(( s2_d * s1_sq - s1_s2 * s1_d) / det_sys, 3.5));
else
    gamma1 = 1;  gamma2 = 1;          % steps (nearly) parallel: plain Adam
end

% Update both networks
for i = 1:numel(fields)
    k = fields{i};
    p1.(k) = p1.(k) + gamma1 * s1.(k);
    p2.(k) = p2.(k) + gamma2 * s2.(k);
end
end