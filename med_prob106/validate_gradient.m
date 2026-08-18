% VALIDATE_GRADIENT.M
%
% Run this FIRST before running Adam.
% It checks that the adjoint gradient matches finite differences.
% If the errors are below ~1e-5, your adjoint is correct.
% If they are large (>1e-2), something is wrong — do not proceed.

clc; clear; close all;

%% ── Setup ──────────────────────────────────────────────────────
prob = setup_truss_10bar();

% Test at a non-uniform design point (more informative than uniform)
a = [15; 5; 20; 8; 12; 3; 25; 10; 7; 18];

fprintf('\n');
fprintf('══════════════════════════════════════════════════\n');
fprintf('  GRADIENT VALIDATION:  Adjoint vs Finite Diff   \n');
fprintf('══════════════════════════════════════════════════\n\n');

%% ── Adjoint gradient ───────────────────────────────────────────
[loss, info] = truss_forward(a, prob);
g_adjoint    = truss_adjoint(a, prob, info);

fprintf('Objective at test point:  %.4f\n', loss);
fprintf('  Weight  = %.4f\n', info.weight);
fprintf('  Penalty = %.4f\n\n', info.penalty);

%% ── Finite difference gradient (central differences) ──────────
eps_fd = 1e-7;
g_fd   = zeros(prob.n_elements, 1);

for i = 1:prob.n_elements
    a_plus     = a;
    a_plus(i)  = a_plus(i) + eps_fd;
    [loss_plus, ~] = truss_forward(a_plus, prob);

    a_minus     = a;
    a_minus(i)  = a_minus(i) - eps_fd;
    [loss_minus, ~] = truss_forward(a_minus, prob);

    g_fd(i) = (loss_plus - loss_minus) / (2 * eps_fd);
end

%% ── Compare ────────────────────────────────────────────────────
fprintf('%-6s %14s %14s %14s\n', 'Bar', 'Adjoint', 'Finite Diff', 'Rel Error');
fprintf('%s\n', repmat('-', 1, 52));

for i = 1:prob.n_elements
    rel_err = abs(g_adjoint(i) - g_fd(i)) / (abs(g_fd(i)) + 1e-12);
    fprintf('%-6d %14.6f %14.6f %14.2e\n', i, g_adjoint(i), g_fd(i), rel_err);
end

max_err = max(abs(g_adjoint - g_fd) ./ (abs(g_fd) + 1e-12));
fprintf('\nMax relative error:  %.2e\n\n', max_err);

if max_err < 1e-4
    fprintf('  ✓  PASSED — Adjoint is correct. Safe to run Adam.\n\n');
else
    fprintf('  ✗  FAILED — Adjoint has errors. Debug before proceeding.\n\n');
end

%% ── Cost comparison ────────────────────────────────────────────
fprintf('Cost comparison (per gradient evaluation):\n');
fprintf('  Finite differences : %d forward solves\n', 2 * prob.n_elements);
fprintf('  Adjoint method     : 1 forward + 1 backward solve\n');
fprintf('  Speedup            : %.0f×\n\n', 2 * prob.n_elements / 2);
