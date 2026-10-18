% Optimization Visualization for Twin Gradient (2D Problems)
function visualize_twin(f, history, w01, w02, w_opt, w_ref)
% This is a function, not a script: it needs the results of an optimizer run.
% Run adam_runner_twin_gradient.m instead of running visualize_twin.m directly.
if nargin < 5
    error('visualize_twin:missingInputs', ...
        ['visualize_twin(f, history, w01, w02, w_opt[, w_ref]) needs the results of a run. ' ...
         'Run your twin runner script instead of running visualize_twin.m directly.']);
end

% w_ref is the reference / theoretical minimum (e.g. from fminsearch).
% If it is not supplied, fall back to the optimizer's own result.
if nargin < 6 || isempty(w_ref)
    w_ref = w_opt;
    ref_name = 'Twin Adam solution';
else
    ref_name = 'Global minimum';
end

% Reshape to column vectors
w01 = w01(:);
w02 = w02(:);
w_ref = w_ref(:);

% The path matrices for both trajectories
w_all1 = [w01, history.w1];
w_all2 = [w02, history.w2];
total_steps = size(w_all1, 2);

% Calculate the function height value history for both twins
f_all1 = zeros(1, total_steps);
f_all2 = zeros(1, total_steps);
for i = 1:total_steps
    f_all1(i) = f(w_all1(:, i));
    f_all2(i) = f(w_all2(:, i));
end

% Setup a grid encompassing both paths and the reference target
pts = [w_all1, w_all2, w_ref];
lo = min(pts, [], 2);
hi = max(pts, [], 2);

% Pad by 20% of the span, with a floor so a flat span does not collapse
span = max(hi - lo, 1e-3);
pad = 0.20 * span;
lo = lo - pad;
hi = hi + pad;

n_grid = 400;
[X1, X2] = meshgrid(linspace(lo(1), hi(1), n_grid), linspace(lo(2), hi(2), n_grid));

% Evaluate the objective function heights across the grid via linear indexing
Z = zeros(size(X1));
for i = 1:numel(X1)
    Z(i) = f([X1(i); X2(i)]);
end

% Smart Scaling Selection
% If the landscape height span is massive, use log scaling. Otherwise, keep it linear.
if (max(Z(:)) - min(Z(:))) > 500
    scale_fn = @(v) sign(v) .* log10(abs(v) + 1);
    z_label_str = 'log_{10}(|f(w)| + 1)';
else
    scale_fn = @(v) v;
    z_label_str = 'z';
end
Z_plot = scale_fn(Z);

% Iteration counter: t = 0 is index 1
t_all = 0:total_steps-1;

% Draw a marker on a subset of steps only, preventing overlapping clutter
mk = unique([1:max(1, round(total_steps/40)):total_steps, total_steps]);

% Legend text carries coordinates
ref_label = sprintf('%s (%.3f, %.3f)', ref_name, w_ref(1), w_ref(2));

%% Plot 1: 2D Contour Pathway Plot
figure('Name', '2D Contour Plot (Twin Adam)', 'NumberTitle', 'off');
hold on;
contour(X1, X2, Z_plot, 40, 'LineWidth', 0.8, 'HandleVisibility', 'off');

% Trajectory paths
plot(w_all1(1,:), w_all1(2,:), 'o-r', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Twin 1 Path');
plot(w_all2(1,:), w_all2(2,:), 's-b', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Twin 2 Path');

% Starting points
plot(w01(1), w01(2), 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start w_{01}');
plot(w02(1), w02(2), 'bs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start w_{02}');

% Target minimum
plot(w_ref(1), w_ref(2), 'g*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', ref_label);
hold off;

set(gca, 'FontSize', 16);
axis([lo(1) hi(1) lo(2) hi(2)]);
xlabel('w1');
ylabel('w2');
title('2D Twin Adam Optimization Pathways');
legend('Location', 'best');
grid on;

%% Plot 2: Convergence Plot
figure('Name', 'Convergence (Twin Adam)', 'NumberTitle', 'off');
if all(f_all1 > 0) && all(f_all2 > 0) && (max([f_all1, f_all2]) / min([f_all1, f_all2])) > 100
    semilogy(t_all, f_all1, 'r-',  'LineWidth', 2, 'DisplayName', 'Twin 1');
    hold on;
    semilogy(t_all, f_all2, 'b--', 'LineWidth', 2, 'DisplayName', 'Twin 2');
    hold off;
    ylabel('Objective Value f(w) [log scale]');
else
    plot(t_all, f_all1, 'r-',  'LineWidth', 2, 'DisplayName', 'Twin 1');
    hold on;
    plot(t_all, f_all2, 'b--', 'LineWidth', 2, 'DisplayName', 'Twin 2');
    hold off;
    ylabel('Objective Value f(w)');
end

set(gca, 'FontSize', 16);
xlabel('t (iterations)');
title('Convergence of Both Trajectories');
legend('Location', 'best');
grid on;

%% Plot 3: 3D Surface
figure('Name', '3D Surface Plot (Twin Adam)', 'NumberTitle', 'off');
hold on;
surf(X1, X2, Z_plot, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'HandleVisibility', 'off');
colormap(jet);
colorbar;

% Lift trajectories slightly off the surface so they remain clearly visible
z_offset = 0.04 * (max(Z_plot(:)) - min(Z_plot(:)));
z_path1 = scale_fn(f_all1) + z_offset;
z_path2 = scale_fn(f_all2) + z_offset;

plot3(w_all1(1,:), w_all1(2,:), z_path1, 'o-r', 'LineWidth', 2.0, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Twin 1 Path');
plot3(w_all2(1,:), w_all2(2,:), z_path2, 's-b', 'LineWidth', 2.0, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Twin 2 Path');

% Plotting the reference point
z_min = scale_fn(f(w_ref)) + z_offset;
plot3(w_ref(1), w_ref(2), z_min, 'g*', 'MarkerSize', 14, 'LineWidth', 3, 'DisplayName', ref_label);
hold off;

set(gca, 'FontSize', 14);
xlabel('w1 (x)');
ylabel('w2 (y)');
zlabel(z_label_str);
title('3D Objective Surface & Twin Paths');
legend('Location', 'best');
view(-45, 45);
grid on;
end