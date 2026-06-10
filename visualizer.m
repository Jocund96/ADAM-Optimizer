function visualizer(cost_func, history, w0, w_opt)
figure('Position', [100, 100, 1200, 500]);

% Plot 1: 2D Contour Trajectory
subplot(1, 2, 1);
[X, Y] = meshgrid(-2:0.1:2, -1:0.1:3);
Z = zeros(size(X));
for i = 1:numel(X)
    Z(i) = cost_func([X(i); Y(i)]);
end
contour(X, Y, Z, 50); hold on;
plot(history.w(1,:), history.w(2,:), 'r.-', 'LineWidth', 1.5);
plot(w0(1), w0(2), 'go', 'MarkerFaceColor', 'g');
plot(w_opt(1), w_opt(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
title('ADAM Optimization Path on Contour Map');
xlabel('x_1'); ylabel('x_2'); grid on;

% Plot 2: Convergence History (Loss Curve)
subplot(1, 2, 2);
loss_history = zeros(size(history.w, 2), 1);
for t = 1:size(history.w, 2)
    loss_history(t) = cost_func(history.w(:, t));
end
semilogy(loss_history, 'b-', 'LineWidth', 2);
title('Convergence Log-Loss Curve');
xlabel('Iteration Iteration (t)'); ylabel('Objective Function Value f(w)');
grid on;
end