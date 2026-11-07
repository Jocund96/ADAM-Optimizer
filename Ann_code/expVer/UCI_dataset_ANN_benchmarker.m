clc; clear; close all;
%% Reproducibility
rng(1); % change the seed to check that results hold across runs

%% Import UCI Data
data = rmmissing(readmatrix("Concrete_Data.xls"));
size(data);

x = transpose(data(:,1:8));
Y = transpose(data(:, 9));
size(x);
size(Y);

N = size(x,2);
idx = randperm(N);
nTest = round(0.2*N);
idxTest = idx(1:nTest);
idxTrain = idx(nTest+1:end);

%% The configuration for the ANN
config = []; % struct that carries the configuration for the ANN

config.layers = [size(x, 1), 16, 1];  %[input_dim, hidden_dim, output_dim]
config.num_epochs = 10000;
config.hp.learning_rate = 0.001;
config.activation = 'tanh'; %tanh, relu, sigmoid, leakyrelu, elu, softplus, swish, linear
config.hp.batch_size = 'N'; % an number for mini batch, 1 for pure SGD and N for BGD
config.optimizer = @adam_optimizer_step;


%% Call the function to train the ANN
[results, losses] = ADAM_ANN(x(:,idxTrain), Y(idxTrain), config);


%% Call Training function
fprintf('Running the benchmarker.... \n ');

% Registry of optimizers to benchmark.
ann_runners = {
    struct('name', 'Adam',    'fn', @(x, Y, config) ADAM_ANN(x(:,idxTrain), Y(idxTrain), config))
    struct('name', 'AdamW',    'fn', @(x, Y, config) ADAM_ANN(x(:,idxTrain), Y(idxTrain), setfield(config, 'optimizer', @adam_optimizer_W_step)))
    struct('name', 'Adam AMSGrad',    'fn', @(x, Y, config) ADAM_ANN(x(:,idxTrain), Y(idxTrain), setfield(config, 'optimizer', @adam_optimizer_AMSGrad_step)))
    struct('name', 'Adam Twin', 'fn', @(x, Y, config) ADAM_ANN_twin(x(:,idxTrain), Y(idxTrain), config))
    };

nOpt   = size(ann_runners, 1);
colors = lines(nOpt);                 % one color per optimizer

bench_results = struct('ANN_Runner', {}, 'Layers', {}, 'Time_total_s', {},'Time_epoch_s', {}, ...
    'Train_MSE', {},'Best_Val_MSE',{} , 'Test_MSE', {}, 'G', {}, 'Grad_evals',{});

figure('Name', 'Concrete data Optimizer Comparison', 'Position', [100, 100, 1500, 450]);

%loop through all the ANN Runners
for k = 1:numel(ann_runners)
    runner = ann_runners{k};
    rng(2);  % same random numbers for all the runners
    tic;
    [results,losses] = runner.fn(x, Y, config);
    elapsed = toc;

    bench_results(k) = benchmark_row(runner.name, elapsed, results, losses);

    % Validation Loss Plot
    subplot(1, 3, 1); hold on; grid on;
    semilogy(losses.valLoss, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (Min: %.4e)', runner.name, min(losses.valLoss)));

    % Train Loss Plot
    subplot(1, 3, 2); hold on; grid on;
    semilogy(losses.trainLoss, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (Min: %.4e)', runner.name, min(losses.trainLoss)));

    % Validation loss over training time (equal computing cost)
    subplot(1, 3, 3); hold on; grid on;
    semilogy(results.time, losses.valLoss, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', runner.name);

end

fprintf('\n');
format short e
disp(struct2table(bench_results));
format short


% Formatting Subplot 1 (Validation)
subplot(1, 3, 1);
set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Epoch'); ylabel('Validation MSE (Normalized)');
title('Validation Loss Convergence');
legend('show', 'Location', 'northeast');

% Formatting Subplot 2 (Train)
subplot(1, 3, 2);
set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Epoch'); ylabel('Train MSE (Normalized)');
title('Training Loss Convergence');
legend('show', 'Location', 'northeast');

% Formatting Subplot 3 (Validation over time)
subplot(1, 3, 3);
set(gca, 'YScale', 'log', 'FontSize', 11);
xlabel('Training time [s]'); ylabel('Validation MSE (Normalized)');
title('Validation Loss over Time');
legend('show', 'Location', 'northeast');


% Computes one runner's summary metrics as a row for the comparison table
function row = benchmark_row(name, elapsed, results, losses)
row.ANN_Runner   = string(name);
row.Layers       = mat2str(results.layers);
row.Time_total_s = elapsed;                                   % total time taken in the training
row.Time_epoch_s = results.time(end)/results.num_epochs;      % mean time per epoch
row.Train_MSE    = losses.meanTrainLoss;
row.Best_Val_MSE = min(losses.valLoss);                       % best validation MSE
row.Test_MSE     = losses.meanTestLoss;
row.G            = losses.meanTestLoss/losses.meanTrainLoss;  % gap ratio (abs not needed, both > 0)
row.Grad_evals   = results.n_updates;                            % cost > the number of updates (for benchmarking Twin ADAM)
end