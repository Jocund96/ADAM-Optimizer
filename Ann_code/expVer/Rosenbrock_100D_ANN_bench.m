clc; clear; close all;

%% Reproducibility
rng(1); % change the seed to check that results hold across runs

%% Generate 100D Extended Rosenbrock Data
D = 100;      % 100 dimensions
N = 100000;     % 100000 sample points
x = -2 + 4 * rand(D, N); % 100 x 4000 matrix

% Vectorized 100D Rosenbrock calculation across rows 1..99 and 2..100
X_curr = x(1:end-1, :);
X_next = x(2:end, :);
Y = sum(100 * (X_next - X_curr.^2).^2 + (1 - X_curr).^2, 1); % 1 x 4000

%% Data Shuffling
shuffle_idx = randperm(N);
x = x(:, shuffle_idx);
Y = Y(shuffle_idx);

%% Data Normalization (Z-score)
x_mean = mean(x, 2);
x_std  = std(x, 0, 2);
x_norm = (x - x_mean) ./ x_std;

Y_mean = mean(Y);
Y_std  = std(Y);
Y_norm = (Y - Y_mean) / Y_std;

%% Compare Optimizers on 100D Problem
optimizers = {@adam_optimizer_step, @adam_optimizer_AMSGrad_step};
opt_names  = {'Standard Adam', 'AMSGrad'};
colors     = {[0, 0.447, 0.741], [0.85, 0.325, 0.098]}; % Blue, Red

exec_times = zeros(1, length(optimizers));

figure('Name', '100D Optimizer Comparison (Adam vs AMSGrad)', 'Position', [100, 100, 1100, 450]);

for k = 1:length(optimizers)
    rng(1); % Reset seed so every optimizer gets the exact same data split & init weights!
    
    config.layers = [D, 128, 64, 1]; 
    config.num_epochs = 2000;
    config.hp.learning_rate = 0.002;
    config.activation = 'leakyrelu';    
    config.optimizer = optimizers{k};
    
    fprintf('\n Running %s on 100D Rosenbrock\n', opt_names{k});
    % Measure training time
    tic;
    [results, losses] = ADAM_ANN(x_norm, Y_norm, config);
    exec_times(k) = toc;

    % Validation Loss Plot
    subplot(1, 2, 1); hold on; grid on;
    semilogy(losses.valLoss, 'Color', colors{k}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (Min: %.4e)', opt_names{k}, min(losses.valLoss)));
    
    % Train Loss Plot
    subplot(1, 2, 2); hold on; grid on;
    semilogy(losses.trainLoss, 'Color', colors{k}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (Min: %.4e)', opt_names{k}, min(losses.trainLoss)));
end

% Formatting Subplot 1 (Validation)
subplot(1, 2, 1);
set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Epoch'); ylabel('Validation MSE (Normalized)');
title('Validation Loss Convergence');
legend('show', 'Location', 'northeast');

% Formatting Subplot 2 (Train)
subplot(1, 2, 2);
set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Epoch'); ylabel('Train MSE (Normalized)');
title('Training Loss Convergence');
legend('show', 'Location', 'northeast');

fprintf('\n TIMING RESULTS\n');
for k = 1:length(optimizers)
    fprintf('%-15s | Time: %6.2f sec | Min Val MSE: %.4e\n', ...
        opt_names{k}, exec_times(k), min(losses.valLoss));
end


