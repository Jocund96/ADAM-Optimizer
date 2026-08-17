% Optimization Visualization for 100D Problems
function visualize_100d(f, g, history, w0, ~)

% The path matrix
w_all = [w0, history.w];
total_steps = size(w_all, 2);

% Calculate the function height value history and gradient norm history
f_all = zeros(1, total_steps);
grad_norm_all = zeros(1, total_steps);

for i = 1:total_steps
    f_all(i) = f(w_all(:, i));
    grad_norm_all(i) = norm(g(w_all(:, i)));
end

D = length(w0);

%% Plot 1: Objective Value Convergence
figure('Name', 'Convergence f(w)', 'NumberTitle', 'off')
semilogy(0:total_steps-1, f_all, 'LineWidth', 2, 'Color', [0.85, 0.325, 0.098]);
set(gca, 'FontSize', 16);
xlabel('t (iterations)');
ylabel('Objective Value f(w) [log scale]');
title(sprintf('Convergence for D = %d', D));
grid on

%% Plot 2: Gradient Norm Convergence
figure('Name', 'Gradient Norm Convergence', 'NumberTitle', 'off')
semilogy(0:total_steps-1, grad_norm_all, 'LineWidth', 2, 'Color', [0, 0.447, 0.741]);
set(gca, 'FontSize', 16);
xlabel('t (iterations)');
ylabel('||\nabla f(w)||_2 [log scale]');
title('Gradient Norm Decay');
grid on

%% Plot 3: Trajectory of Selected Dimensions
figure('Name', 'Selected Coordinate Trajectories', 'NumberTitle', 'off')
hold on
plot(0:total_steps-1, w_all(1, :), 'LineWidth', 2, 'DisplayName', 'w_1');
plot(0:total_steps-1, w_all(2, :), 'LineWidth', 2, 'DisplayName', 'w_2');
plot(0:total_steps-1, w_all(50, :), 'LineWidth', 2, 'DisplayName', 'w_{50}');
plot(0:total_steps-1, w_all(100, :), 'LineWidth', 2, 'DisplayName', 'w_{100}');
yline(1.0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Target (1.0)');
hold off
set(gca, 'FontSize', 16);
xlabel('t (iterations)');
ylabel('Parameter Value');
title('Evolution of Selected Design Variables over Time');
legend('Location', 'best');
grid on
end