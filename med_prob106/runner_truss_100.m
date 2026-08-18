% RUNNER_TRUSS_100.M
%
% Optimise a grid truss with ~100 design variables using Adam + adjoint.
%
% The ONLY thing that changed from runner_truss.m is this line:
%
%   prob = setup_truss_10bar();    →    prob = setup_truss_grid(6, 4);
%
% Everything else — truss_forward, truss_adjoint, truss_gradient,
% adam_optimizer — is IDENTICAL.  Not one line changed.
% ──────────────────────────────────────────────────────────────

clc; clear; close all;

%% ── Problem setup ─────────────────────────────────────────────
%  6 bays wide × 4 bays tall = 106 bars (design variables)
%  Change these two numbers to get more or fewer variables:
%    (5, 4) →  89 bars
%    (6, 4) → 106 bars
%    (7, 4) → 123 bars
%    (8, 3) → 107 bars

prob = setup_truss_grid(6, 4);

% Starting point: uniform areas
a0 = 10 * ones(prob.n_elements, 1);

% Initial evaluation
[loss0, info0] = truss_forward(a0, prob);

fprintf('══════════════════════════════════════════════════\n');
fprintf('  GRID TRUSS OPTIMISATION  —  %d DESIGN VARIABLES\n', prob.n_elements);
fprintf('══════════════════════════════════════════════════\n\n');
fprintf('Initial weight:    %.2f lbs\n', info0.weight);
fprintf('Initial penalty:   %.2f\n',     info0.penalty);
fprintf('Initial loss:      %.2f\n\n',   loss0);

%% ── Validate gradient before running ──────────────────────────
fprintf('Validating adjoint gradient (spot check on 5 random bars)...\n');
[~, info_v] = truss_forward(a0, prob);
g_adj = truss_adjoint(a0, prob, info_v);

eps_fd    = 1e-7;
check_ids = sort(randperm(prob.n_elements, min(5, prob.n_elements)));
max_err   = 0;

for idx = 1:length(check_ids)
    i = check_ids(idx);
    ap = a0; ap(i) = ap(i) + eps_fd;
    am = a0; am(i) = am(i) - eps_fd;
    [lp, ~] = truss_forward(ap, prob);
    [lm, ~] = truss_forward(am, prob);
    g_fd_i  = (lp - lm) / (2 * eps_fd);
    rel_err = abs(g_adj(i) - g_fd_i) / (abs(g_fd_i) + 1e-12);
    max_err = max(max_err, rel_err);
    fprintf('  Bar %3d:  adjoint=%.6f  FD=%.6f  err=%.2e\n', ...
            i, g_adj(i), g_fd_i, rel_err);
end

if max_err < 1e-4
    fprintf('  Gradient OK (max error: %.2e)\n\n', max_err);
else
    fprintf('  WARNING: gradient error %.2e — check adjoint!\n\n', max_err);
end

%% ── Build gradient function and run Adam ──────────────────────
grad_f = @(a) truss_gradient(a, prob);

% Adam hyperparameters
max_iter  = 3000;
alpha     = 0.05;
beta1     = 0.9;
beta2     = 0.999;
epsilon   = 1e-8;
tolerance = 1e-6;

fprintf('Running Adam  (alpha=%.3f, max_iter=%d)...\n', alpha, max_iter);

tic;
[a_opt, history] = adam_optimizer(grad_f, a0, max_iter, ...
                                  alpha, beta1, beta2, epsilon, tolerance);
elapsed = toc;

% Clamp to bounds
a_opt = max(a_opt, prob.a_min);
a_opt = min(a_opt, prob.a_max);

%% ── Final evaluation ──────────────────────────────────────────
[loss_final, info_final] = truss_forward(a_opt, prob);
n_iters = size(history.w, 2);

fprintf('\n── RESULTS ──────────────────────────────────────\n\n');
fprintf('  Time:             %.1f seconds\n',  elapsed);
fprintf('  Iterations:       %d\n',            n_iters);
fprintf('  Initial weight:   %.2f lbs\n',      info0.weight);
fprintf('  Optimised weight: %.2f lbs\n',      info_final.weight);
fprintf('  Weight reduction: %.1f%%\n',        100*(1 - info_final.weight/info0.weight));
fprintf('  Final penalty:    %.4f\n',          info_final.penalty);
fprintf('  Max |stress|:     %.2f / %.2f ksi\n', max(abs(info_final.stress)), prob.sigma_max);

% Count active vs near-minimum bars
n_active  = sum(a_opt > prob.a_min + 0.5);
n_removed = sum(a_opt < prob.a_min + 0.5);
fprintf('  Active bars:      %d / %d\n',   n_active,  prob.n_elements);
fprintf('  Near-minimum:     %d / %d\n\n', n_removed, prob.n_elements);

%% ── Compute loss history (subsample for speed) ────────────────
sample_step = max(1, floor(n_iters / 200));   % at most 200 points
samples     = 1:sample_step:n_iters;
loss_hist   = zeros(length(samples), 1);
weight_hist = zeros(length(samples), 1);

for k = 1:length(samples)
    t   = samples(k);
    a_t = max(min(history.w(:,t), prob.a_max), prob.a_min);
    [loss_hist(k), inf_t] = truss_forward(a_t, prob);
    weight_hist(k) = inf_t.weight;
end

%% ── Plots ─────────────────────────────────────────────────────
figure('Name', 'Grid Truss Optimisation', 'Position', [50 50 1200 800]);

% 1. Loss convergence
subplot(2,3,1);
semilogy(samples, loss_hist, 'b-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Loss (log)');
title('Total Loss'); grid on;

% 2. Weight convergence
subplot(2,3,2);
plot(samples, weight_hist, 'r-', 'LineWidth', 2);
xlabel('Iteration'); ylabel('Weight (lbs)');
title('Structural Weight'); grid on;

% 3. Area distribution (histogram)
subplot(2,3,3);
histogram(a_opt, 20, 'FaceColor', [0.2 0.5 0.9], 'EdgeColor', 'white');
xlabel('Bar area (in²)'); ylabel('Count');
title('Area Distribution'); grid on;
xline(prob.a_min, 'r--', 'Min', 'LineWidth', 1.5);

% 4. Stress utilisation histogram
subplot(2,3,4);
util = abs(info_final.stress) / prob.sigma_max * 100;
histogram(util, 20, 'FaceColor', [0.9 0.5 0.2], 'EdgeColor', 'white');
xlabel('Stress utilisation (%)'); ylabel('Count');
title('Stress Utilisation'); grid on;
xline(100, 'r--', 'Limit', 'LineWidth', 1.5);

% 5. Truss visualisation — initial (all bars same thickness)
subplot(2,3,5);
plot_truss(prob, a0, 'Initial Design (uniform)');

% 6. Truss visualisation — optimised (thickness = area)
subplot(2,3,6);
plot_truss(prob, a_opt, 'Optimised Design');

sgtitle(sprintf('Grid Truss — %d Design Variables — Adam + Adjoint', ...
    prob.n_elements), 'FontSize', 14, 'FontWeight', 'bold');


%% ── Local function: plot truss ────────────────────────────────
function plot_truss(prob, a, ttl)
    hold on; axis equal; axis off;
    title(ttl, 'FontSize', 11);

    % Normalise line widths to [0.5, 5]
    a_norm = (a - min(a)) / (max(a) - min(a) + 1e-12);
    lw     = 0.5 + 4.5 * a_norm;

    % Draw bars with thickness proportional to area
    for e = 1:prob.n_elements
        ni = prob.elements(e, 1);
        nj = prob.elements(e, 2);
        x  = [prob.nodes(ni,1), prob.nodes(nj,1)];
        y  = [prob.nodes(ni,2), prob.nodes(nj,2)];

        % Color: blue for thick (structural), light gray for thin (removed)
        if a(e) < prob.a_min + 0.5
            col = [0.85 0.85 0.85];
        else
            col = [0.15, 0.35 + 0.4*a_norm(e), 0.85];
        end
        plot(x, y, '-', 'LineWidth', lw(e), 'Color', col);
    end

    % Draw nodes
    plot(prob.nodes(:,1), prob.nodes(:,2), 'ko', ...
         'MarkerSize', 3, 'MarkerFaceColor', 'k');

    % Mark fixed nodes
    for d = prob.fixed_dofs(1:2:end)
        node_id = ceil(d / 2);
        plot(prob.nodes(node_id,1), prob.nodes(node_id,2), ...
             'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    end
end

end
