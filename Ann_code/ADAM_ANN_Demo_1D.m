clc; clear; close all;

%% ====== Reproducibility
rng(1);   % change the seed to check that results hold across runs

%% ====== Data
x = 1:1:200;
% ---- the function (change this line to learn any 1D function)
f = @(x) sin(0.2*x);
Y = f(x) + 0.05*randn(1,200);

%% ==== Hyper-Parameters
layers = [1, 40, 1];  %[input_dim, hidden_dim, output_dim]
num_epochs = 20000;
learning_rate = 0.01;
activation = 'relu'; %tanh, relu

%% Call the function to train the ANN
[results, losses] = ADAM_ANN(x, Y, layers, num_epochs, activation, learning_rate);

%% === The Plots for data visualization
Visualize_1D(x,Y, layers ,activation, results, losses);


%% The plotting function
function Visualize_1D(x,Y, layers ,activation, results, losses)

figure
plot(x, Y, 'b.', 'MarkerSize', 10); hold on

plot(results.xFine, results.yFine, 'r-', 'LineWidth', 2);
legend('data', 'ANN'); grid on
title(sprintf('ANN %s, %s   (scaled test MSE = %.4f)', mat2str(layers), activation, losses.testLoss))

figure
semilogy(losses.trainLoss, 'b'); hold on
semilogy(losses.valLoss, 'r');
legend('train', 'validation'); grid on
xlabel('Epoch'); ylabel('MSE (scaled)'); title('Loss curves')
end

