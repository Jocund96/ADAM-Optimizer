function ADAM_ANN_Tanh
clc;clear; close all;
% c) T.Lahmer, 2026

%  %%%%%%%%%%%% Data
x = 1:1:200;

%the function
Y = sin(0.2*x) + 0.05*randn(1,200);

layers = [1, 40, 1];  %[input_dim, hidden_dim, output_dim]
num_epochs = 20000;
learning_rate = 0.01;

% Scale x to [-1, 1] (always do this; raw x up to 200 would saturate tanh)
xMin = min(x);  xMax = max(x);
X   = 2*(x - xMin)/(xMax - xMin) - 1;

params = train_adam(X, Y, layers, num_epochs, learning_rate);

[yPred, cache] = forward(X, params);

figure
plot(x, Y, 'b.', 'MarkerSize', 8);        % FIX: plot over original x (optional)
hold on                                   % FIX: without this the red line erased the blue one
plot(x, yPred, 'r-', 'LineWidth', 2);
legend('data', 'ANN'); grid on
title('my first trained NN')

figure
plot(x, yPred - Y)
title('residuals'); grid on


end


function [params] = train_adam(X, Y, layers, num_epochs, learning_rate)
% X: Eingabedaten (features x samples)  (inputs)
% Y: Zielwerte (outputs x samples)  (outputs)
% layers: z.B. [input_dim, hidden_dim, output_dim]

% Initialisierung
params = init_params(layers);

% ADAM Parameter
beta1 = 0.9;
beta2 = 0.999;
eps = 1e-8;

% Initialisiere Moment-Schätzer  (Moment-estimator)
[m, v] = init_adam(params);

t = 0; % timestep

for epoch = 1:num_epochs
    t = t + 1;

    % Vorwärtspropagation
    [A, cache] = forward(X, params);

    % Verlust (MSE)
    loss = mean((A - Y).^2);

    % Rückwärtspropagation
    grads = backward(X, Y, params, cache);

    % ADAM Update
    [params, m, v] = adam_update(params, grads, m, v, t, learning_rate, beta1, beta2, eps);

    if mod(epoch,100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end
end

%% Initialisierung der Parameter
function params = init_params(layers)
for i = 1:length(layers)-1
    %params.(['W' num2str(i)]) = randn(layers(i+1), layers(i)) * 0.1;
    %params.(['b' num2str(i)]) = zeros(layers(i+1),1);
    params.(['W' num2str(i)]) = randn(layers(i+1), layers(i));   % no * 0.01
    params.(['b' num2str(i)]) = randn(layers(i+1), 1);           % random, not zeros
end
end

%% Initialisierung ADAM Variablen
function [m, v] = init_adam(params)
fields = fieldnames(params);
for i = 1:length(fields)
    m.(fields{i}) = zeros(size(params.(fields{i})));
    v.(fields{i}) = zeros(size(params.(fields{i})));
end
end

%% Vorwärtspropagation
function [A, cache] = forward(X, params)
W1 = params.W1; b1 = params.b1;
W2 = params.W2; b2 = params.b2;

Z1 = W1 * X + b1;
A1 = tanh(Z1); % tanh

Z2 = W2 * A1 + b2;
A = Z2; % lineare Ausgabe

cache = struct('X',X,'Z1',Z1,'A1',A1);
end

%% Rückwärtspropagation
function grads = backward(X, Y, params, cache)

W2 = params.W2;

A1 = cache.A1;
A = params.W2 * A1 + params.b2;

%dZ2 = 2*(A - Y) / m;
dZ2 = 2*(A - Y) / numel(Y);
grads.W2 = dZ2 * A1';
grads.b2 = sum(dZ2,2);

dA1 = W2' * dZ2;
dZ1 = dA1 .* (1 - A1.^2); % tanh Ableitung

grads.W1 = dZ1 * X';
grads.b1 = sum(dZ1,2);
end

%% ADAM Update
function [params, m, v] = adam_update(params, grads, m, v, t, lr, beta1, beta2, eps)
fields = fieldnames(params);

for i = 1:length(fields)
    key = fields{i};

    % 1. Moment (Mean)
    m.(key) = beta1 * m.(key) + (1 - beta1) * grads.(key);

    % 2. Moment (Varianz)
    v.(key) = beta2 * v.(key) + (1 - beta2) * (grads.(key).^2);

    % Bias-Korrektur
    m_hat = m.(key) / (1 - beta1^t);
    v_hat = v.(key) / (1 - beta2^t);

    % Update
    params.(key) = params.(key) - lr * m_hat ./ (sqrt(v_hat) + eps);
end
end
