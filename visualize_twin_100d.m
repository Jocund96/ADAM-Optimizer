% Optimization Visualization for Twin Gradient (100D Problems)
function visualize_twin_100d(f, g, history, w01, w02)
% Format inputs as column vectors
w01 = w01(:);
w02 = w02(:);

% Support both single-run history (.w) and twin history (.w1, .w2)
if isfield(history, 'w1') && isfield(history, 'w2')
    is_twin = true;
    w_all1 = [w01, history.w1];
    w_all2 = [w02, history.w2];
    total_steps = size(w_all1, 2);
else
    is_twin = false;
    w_all1 = [w01, history.w];
    total_steps = size(w_all1, 2);
end

% Preallocate metric arrays
f_all1 = zeros(1, total_steps);
grad_norm1 = zeros(1, total_steps);

if is_twin
    f_all2 = zeros(1, total_steps);
    grad_norm2 = zeros(1, total_steps);
    twin_dist = zeros(1, total_steps);
end

% Compute function values and gradient norms across steps
for i = 1:total_steps
    f_all1(i) = f(w_all1(:, i));
    grad_norm1(i) = norm(g(w_all1(:, i)));
    
    if is_twin
        f_all2(i) = f(w_all2(:, i));
        grad_norm2(i) = norm(g(w_all2(:, i)));
        twin_dist(i) = norm(w_all1(:, i) - w_all2(:, i));
    end
end

iters = 0:total_steps-1;
D = length(w01);

%% Plot 1: Objective Value Convergence
figure('Name', 'Convergence f(w)', 'NumberTitle', 'off');
if is_twin
    semilogy(iters, f_all1, 'LineWidth', 2, 'Color', [0.85, 0.325, 0.098], 'DisplayName', 'Twin 1');
    hold on;
    semilogy(iters, f_all2, '--', 'LineWidth', 2, 'Color', [0.494, 0.184, 0.556], 'DisplayName', 'Twin 2');
    hold off;
    legend('Location', 'best');
else
    semilogy(iters, f_all1, 'LineWidth', 2, 'Color', [0.85, 0.325, 0.098]);
end
set(gca, 'FontSize', 14);
xlabel('t (iterations)');
ylabel('Objective Value f(w) [log scale]');
title(sprintf('Objective Value Decay (D = %d)', D));
grid on;

%% Plot 2: Gradient Norm Convergence
figure('Name', 'Gradient Norm Convergence', 'NumberTitle', 'off');
if is_twin
    semilogy(iters, grad_norm1, 'LineWidth', 2, 'Color', [0, 0.447, 0.741], 'DisplayName', '||\nabla f(w_1)||_2');
    hold on;
    semilogy(iters, grad_norm2, '--', 'LineWidth', 2, 'Color', [0.929, 0.694, 0.125], 'DisplayName', '||\nabla f(w_2)||_2');
    hold off;
    legend('Location', 'best');
else
    semilogy(iters, grad_norm1, 'LineWidth', 2, 'Color', [0, 0.447, 0.741]);
end
set(gca, 'FontSize', 14);
xlabel('t (iterations)');
ylabel('||\nabla f(w)||_2 [log scale]');
title('Gradient Norm Decay');
grid on;

%% Plot 3: Trajectory of Selected Dimensions
figure('Name', 'Selected Coordinate Trajectories', 'NumberTitle', 'off');
hold on;
if is_twin
    % Twin 1 coordinates (solid lines)
    plot(iters, w_all1(1, :), '-',  'LineWidth', 2, 'Color', [0.0, 0.45, 0.74], 'DisplayName', 'w_1^{(1)}');
    plot(iters, w_all1(2, :), '-',  'LineWidth', 2, 'Color', [0.85, 0.33, 0.10], 'DisplayName', 'w_2^{(1)}');
    plot(iters, w_all1(50, :), '-', 'LineWidth', 2, 'Color', [0.47, 0.67, 0.19], 'DisplayName', 'w_{50}^{(1)}');
    plot(iters, w_all1(D, :), '-',  'LineWidth', 2, 'Color', [0.49, 0.18, 0.56], 'DisplayName', sprintf('w_{%d}^{(1)}', D));
    
    % Twin 2 coordinates (dashed lines)
    plot(iters, w_all2(1, :), '--',  'LineWidth', 1.8, 'Color', [0.0, 0.45, 0.74], 'DisplayName', 'w_1^{(2)}');
    plot(iters, w_all2(2, :), '--',  'LineWidth', 1.8, 'Color', [0.85, 0.33, 0.10], 'DisplayName', 'w_2^{(2)}');
    plot(iters, w_all2(50, :), '--', 'LineWidth', 1.8, 'Color', [0.47, 0.67, 0.19], 'DisplayName', 'w_{50}^{(2)}');
    plot(iters, w_all2(D, :), '--',  'LineWidth', 1.8, 'Color', [0.49, 0.18, 0.56], 'DisplayName', sprintf('w_{%d}^{(2)}', D));
else
    plot(iters, w_all1(1, :), 'LineWidth', 2, 'DisplayName', 'w_1');
    plot(iters, w_all1(2, :), 'LineWidth', 2, 'DisplayName', 'w_2');
    plot(iters, w_all1(50, :), 'LineWidth', 2, 'DisplayName', 'w_{50}');
    plot(iters, w_all1(D, :), 'LineWidth', 2, 'DisplayName', sprintf('w_{%d}', D));
end
yline(1.0, ':k', 'LineWidth', 1.5, 'DisplayName', 'Target (1.0)');
hold off;
set(gca, 'FontSize', 14);
xlabel('t (iterations)');
ylabel('Parameter Value');
title('Selected Design Variables over Time (Solid: Twin 1, Dashed: Twin 2)');
legend('Location', 'best');
grid on;

%% Plot 4: Inter-Twin Distance ||w1 - w2||_2 (Twin Metric)
if is_twin
    figure('Name', 'Twin Distance Convergence', 'NumberTitle', 'off');
    semilogy(iters, twin_dist, 'LineWidth', 2, 'Color', [0.635, 0.078, 0.184]);
    set(gca, 'FontSize', 14);
    xlabel('t (iterations)');
    ylabel('||w_1 - w_2||_2 [log scale]');
    title('Separation Distance Between Twins');
    grid on;
end
end