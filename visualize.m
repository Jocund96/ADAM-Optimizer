% Optimization Visualization
function visualize(f, history, w0, w_opt)

% The path matrix
w_all = [w0, history.w];

% Calculate the function height value history 
f_all = zeros(1, size(w_all, 2));
for i = 1:size(w_all, 2)
    f_all(i) = f(w_all(:, i));
end

% Setup a grid
% for Beale
[X1, X2] = meshgrid(linspace(-1, 4.5, 500), linspace(-1.5, 1.5, 500)); 
% for Rosenbrock
%[X1, X2] = meshgrid(linspace(-2, 2, 1000), linspace(-1, 3, 1000));

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

%% Plot 1: 2D Contour Pathway Plot
figure('Name', '2D Contour Plot', 'NumberTitle', 'off')
hold on
contour(X1, X2, Z_plot, 40, 'LineWidth', 0.8, 'HandleVisibility', 'off') 
plot(w_all(1,:), w_all(2,:), 'o-r', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Adam Path')
plot(w_opt(1), w_opt(2), 'g*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Global Minimum (3.0, 0.5)')
hold off
set(gca, 'FontSize', 16);
xlabel('w1');
ylabel('w2');
title('2D Adam Optimization Path');
legend('Location', 'best');
grid on

%% Plot 2: Convergence Plot
figure('Name', 'Convergence', 'NumberTitle', 'off')
plot(f_all, 'LineWidth', 2);
set(gca, 'FontSize', 16);
xlabel('t (iterations)');
ylabel('Objective Value f(w)');
title('Convergence');
grid on

%% PLOT 3: 3D Surface
figure('Name', '3D Surface Plot', 'NumberTitle', 'off')
hold on
surf(X1, X2, Z_plot, 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'HandleVisibility', 'off')
colormap(jet)
colorbar

z_path = scale_fn(f_all);
plot3(w_all(1,:), w_all(2,:), z_path, 'o-r', 'LineWidth', 2.5, 'MarkerSize', 4, 'DisplayName', 'Adam Path')

% Plotting the green star point for the minimum
z_min = scale_fn(f(w_opt));
plot3(w_opt(1), w_opt(2), z_min, 'g*', 'MarkerSize', 14, 'LineWidth', 3, 'DisplayName', 'Global Minimum')
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