clc; clear; close all;

%% Reproducibility
rng(1);   % change the seed to check that results hold across runs

%% Generate 2D Rosenbrock Data
N = 1000;
x = -2 + 4 * rand(2, N); % 2 x 1000 uniform samples in [-2, 2]^2

rosenbrock_2d = @(X) (1 - X(1,:)).^2 + 100 * (X(2,:) - X(1,:).^2).^2;
Y = rosenbrock_2d(x);    % 1 x 1000 target values

%% Configure ANN 
config.layers = [2, 32, 1];
config.num_epochs = 4000;
config.hp.learning_rate = 0.01;
config.activation = 'tanh';
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
