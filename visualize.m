% Optimization Visualization
function visualize(f, history, w0, w_opt, w_ref)

% This is a function, not a script: it needs the results of an optimizer run.
% Run one of the adam_runner_*.m scripts, do not run this file on its own.
if nargin < 4
    error('visualize:missingInputs', ...
        ['visualize(f, history, w0, w_opt[, w_ref]) needs the results of a run. ' ...
         'Run adam_runner_rosenbrock_func or adam_runner_Beale_func instead of running visualize.m directly.']);
end

% w_ref is the reference / theoretical minimum (e.g. from fminsearch).
% If it is not supplied, fall back to Adam's own result.
if nargin < 5 || isempty(w_ref)
    w_ref = w_opt;
    ref_name = 'Adam solution';
else
    ref_name = 'Global minimum';
end

% The path matrix
w_all = [w0, history.w];

% Calculate the function height value history
f_all = zeros(1, size(w_all, 2));
for i = 1:size(w_all, 2)
    f_all(i) = f(w_all(:, i));
end

% Setup a grid
% The window is derived from the trajectory itself, so the whole path and
% both marked points are always inside the plot regardless of the problem.
pts = [w_all, w_ref];
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

% Iteration counter: f_all(1) is f(w0), i.e. t = 0
t_all = 0:numel(f_all)-1;

% Draw a marker on a subset of steps only, otherwise a long run turns the
% path into a solid blob of overlapping circles
mk = unique([1:max(1,round(numel(f_all)/40)):numel(f_all), numel(f_all)]);

% Legend text carries the actual coordinates instead of a hardcoded pair
ref_label = sprintf('%s (%.3f, %.3f)', ref_name, w_ref(1), w_ref(2));

%% Plot 1: 2D Contour Pathway Plot
figure('Name', '2D Contour Plot', 'NumberTitle', 'off')
hold on
contour(X1, X2, Z_plot, 40, 'LineWidth', 0.8, 'HandleVisibility', 'off')
plot(w_all(1,:), w_all(2,:), 'o-r', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Adam Path')
plot(w0(1), w0(2), 'bs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start w_0')
plot(w_ref(1), w_ref(2), 'g*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', ref_label)
hold off
set(gca, 'FontSize', 16);
axis([lo(1) hi(1) lo(2) hi(2)])
xlabel('w1');
ylabel('w2');
title('2D Adam Optimization Path');
legend('Location', 'best');
grid on

%% Plot 2: Convergence Plot
figure('Name', 'Convergence', 'NumberTitle', 'off')
% Log axis only when it is usable: strictly positive values spanning decades
if all(f_all > 0) && (max(f_all) / min(f_all)) > 100
    semilogy(t_all, f_all, 'LineWidth', 2);
    ylabel('Objective Value f(w) [log scale]');
else
    plot(t_all, f_all, 'LineWidth', 2);
    ylabel('Objective Value f(w)');
end
set(gca, 'FontSize', 16);
xlabel('t (iterations)');
title('Convergence');
grid on

%% PLOT 3: 3D Surface
figure('Name', '3D Surface Plot', 'NumberTitle', 'off')
hold on
surf(X1, X2, Z_plot, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'HandleVisibility', 'off')
colormap(jet)
colorbar

% Lift the path slightly off the surface so it is not swallowed by it
z_offset = 0.04 * (max(Z_plot(:)) - min(Z_plot(:)));

z_path = scale_fn(f_all) + z_offset;
plot3(w_all(1,:), w_all(2,:), z_path, 'o-r', 'LineWidth', 2.5, 'MarkerSize', 5, 'MarkerIndices', mk, 'DisplayName', 'Adam Path')

% Plotting the green star point for the minimum
z_min = scale_fn(f(w_ref)) + z_offset;
plot3(w_ref(1), w_ref(2), z_min, 'g*', 'MarkerSize', 14, 'LineWidth', 3, 'DisplayName', ref_label)
hold off
set(gca, 'FontSize', 14);
xlabel('w1 (x)');
ylabel('w2 (y)');
zlabel(z_label_str);
title('3D Plot');
legend('Location', 'best');
view(-45, 45)
grid on
end
