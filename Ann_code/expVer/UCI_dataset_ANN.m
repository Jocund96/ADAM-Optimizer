clc; clear; close all;
%% Reproducibility
rng(1); % change the seed to check that results hold across runs

%% Import UCI Data
data = rmmissing(readmatrix("Concrete_Data.xls"));
size(data)

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
config.optimizer = @adam_optimizer_step;


%% Call the function to train the ANN
[results, losses] = ADAM_ANN(x(:,idxTrain), Y(idxTrain), config);

[bestVal, bestEpoch] = min(losses.valLoss);

YhatTrain = results.predict(x(:, idxTrain));
YhatTest = results.predict(x(:, idxTest));

r2Train = calcR2(Y(idxTrain), YhatTrain);
r2Test  = calcR2(Y(idxTest), YhatTest);

%% Liner regression for comparison
% !!! to use this part install Statistics and Machine Learning Toolbox
x_tr_tran = transpose(x(:, idxTrain));
x_test_tran = transpose(x(:, idxTest));
Y_tr_tran=transpose(Y(idxTrain));
Y_test_tran = transpose(Y(idxTest));
mdl_tr = fitlm(x_tr_tran, Y_tr_tran);

YhatTrain_lr = predict(mdl_tr, x_tr_tran);
YhatTest_lr = predict(mdl_tr, x_test_tran);

r2Train_lr = calcR2(Y_tr_tran, YhatTrain_lr);
r2Test_lr  = calcR2(Y_test_tran, YhatTest_lr);
fprintf('Linear Regression R2 Train: %.4f, R2 Test %.4f \n', r2Train_lr, r2Test_lr);

%% Print results
fprintf('ADAM R2 Train: %.4f, R2 Test %.4f \n', r2Train, r2Test);
fprintf('Best Validation %.4f \n', bestVal);
fprintf('Best Epoch %.4f \n', bestEpoch);

% Formatting Subplot 1 (Train)
subplot(1, 2, 1);
scatter(Y(idxTest), YhatTest)
hold on;
xlabel('True strength (MPa)'); ylabel('Predicted strength (MPa)');
plot([min(Y) max(Y)], [min(Y) max(Y)], 'r--')
title(sprintf('R2 = %.4f', r2Test));

% Formatting Subplot 2 (Test)
subplot(1, 2, 2);
semilogy(losses.trainLoss)
hold on;
semilogy(losses.valLoss)
grid on

xlabel('Epoch'); ylabel('Scaled MSE');
title('Training Loss Convergence');
legend('Train', 'Val', 'Location', 'northeast');


%% R squared evaluation function
function r2 = calcR2(ytrue, ypred)

    r2 = 1 - sum((ytrue - ypred).^2) / sum((ytrue - mean(ytrue)).^2);

end