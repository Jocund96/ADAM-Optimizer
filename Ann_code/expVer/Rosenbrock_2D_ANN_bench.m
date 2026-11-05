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
config.optimizer = @adam_optimizer_step;

%% Train ANN
[results, losses] = ADAM_ANN(x, Y, config);

%% Visualize True Surface vs. ANN Predicted Surface
[X1, X2] = meshgrid(linspace(-2, 2, 60), linspace(-2, 2, 60));
X_grid = [X1(:)'; X2(:)'];

Y_true_grid = reshape(rosenbrock_2d(X_grid), size(X1));
Y_pred_grid = reshape(results.predict(X_grid), size(X1));

figure('Position', [100, 100, 1100, 450]);
subplot(1, 2, 1);
surf(X1, X2, Y_true_grid, 'EdgeColor', 'none');
title('True 2D Rosenbrock Surface');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x_1, x_2)'); view(-35, 30); colorbar;

subplot(1, 2, 2);
surf(X1, X2, Y_pred_grid, 'EdgeColor', 'none');
title(sprintf('ANN Prediction (Scaled Test MSE = %.5f)', losses.meanTestLoss));
xlabel('x_1'); ylabel('x_2'); zlabel('ANN Output'); view(-35, 30); colorbar;

figure;
semilogy(losses.trainLoss, 'b'); hold on;
semilogy(losses.valLoss, 'r');
legend('Train', 'Validation'); grid on;
xlabel('Epoch'); ylabel('Scaled MSE');
title('2D Rosenbrock Loss Curves');