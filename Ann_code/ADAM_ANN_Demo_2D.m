clc; clear; close all;

%% ====== Data
x = 1:1:200;
% ---- the function
Y = sin(0.2*x) + 0.05*randn(1,200);

%% ==== Hyper-Parameters
layers = [1, 40, 1];  %[input_dim, hidden_dim, output_dim]
num_epochs = 20000;
learning_rate = 0.01;
activation = 'tanh';

%% Call the function to train the ANN
[results, losses] = ADAM_ANN(x, Y, layers, num_epochs, activation, learning_rate);

%% === The Plots for data visualization
Visualize_2D(x,Y, layers ,activation, results, losses);


%% The plotting function
function Visualize_2D(x,Y, layers ,activation, results, losses)

figure
plot(x, Y, 'b.', 'MarkerSize', 10); hold on

plot(results.xFine, results.yFine, 'r-', 'LineWidth', 2);
legend('data', 'ANN'); grid on
title(sprintf('ANN %s, %s   (test MSE = %.4f)', mat2str(layers), activation, losses.testLoss))

figure
semilogy(losses.trainLoss, 'b'); hold on
semilogy(losses.valLoss, 'r');
legend('train', 'validation'); grid on
xlabel('Epoch'); ylabel('MSE'); title('Loss curves')
end

