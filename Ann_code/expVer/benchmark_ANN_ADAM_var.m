% Running the adam benchmarking script

% Clear workspace and command window
clear; clc; close all;

%% ====== Reproducibility
rng(1);   % change the seed to check that results hold across runs

%% ====== Data
x = 1:1:200;
% ---- the function (to learn)
f = @(x) sin(0.2*x);
Y = f(x) + 0.05*randn(1,200); % function + added noise

%% The configuration for the ANN
config = []; % struct that carries the configuration for the ANN

config.layers = [1, 20, 1];  %[input_dim, hidden_dim, output_dim]
config.num_epochs = 20000;
config.hp.learning_rate = 0.01;
config.activation = 'tanh'; %tanh, relu, sigmoid, leakyrelu, elu, softplus, swish, linear
config.hp.batch_size = 'N'; % an number for mini batch, 1 for pure SGD and N for BGD
config.optimizer = @adam_optimizer_step;


%% Call Training function
fprintf('Running the benchmarker.... \n ');

% Registry of optimizers to benchmark.
ann_runners = {
    struct('name', 'Adam',    'fn', @(x, Y, config) ADAM_ANN(x, Y, config))
    struct('name', 'AdamW',    'fn', @(x, Y, config) ADAM_ANN(x, Y, setfield(config, 'optimizer', @adam_optimizer_W_step)))
    struct('name', 'Adam AMSGrad',    'fn', @(x, Y, config) ADAM_ANN(x, Y, setfield(config, 'optimizer', @adam_optimizer_AMSGrad_step)))
    struct('name', 'Adam Twin', 'fn', @(x, Y, config) ADAM_ANN_twin(x, Y, config))
    };


bench_results = struct('ANN_Runner', {}, 'Layers', {}, 'Time_total_s', {},'Time_epoch_s', {}, ...
                  'Train_MSE', {},'Best_Val_MSE',{} , 'Test_MSE', {}, 'G', {}, 'Grad_evals',{});

%loop through all the ANN Runners
for k = 1:numel(ann_runners)
    runner = ann_runners{k};
    rng(2);  % same random numbers for all the runners
    tic;
    [results,losses] = runner.fn(x, Y, config);
    elapsed = toc;

    bench_results(k) = benchmark_row(runner.name, elapsed, results, losses);
end

fprintf('\n');
format short e
disp(struct2table(bench_results));
format short



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

%%% Notes
% G ≈ 1: generalizes well.
%G ≈ 2–3: mild overfitting, and common with small data sets.
%G ≫ 3: clear overfitting.