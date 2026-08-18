%Initialize the problem
%prob = setup_truss_10bar();     %  10 variables
prob = setup_truss_grid(6, 4);  % 106 variables

% Starting point:  uniform areas of 10 in² for all bars
a0 = 10 * ones(prob.n_elements, 1);

% Compute initial objective
[loss0, info0] = truss_forward(a0, prob);

%% Build the gradient function handle for ADAM
grad_f = @(a) truss_gradient(a, prob);

%% Run Adam
max_iter  = 25000;
alpha     = 0.05;    
beta1     = 0.9;
beta2     = 0.999;
epsilon   = 1e-8;
tolerance = 1e-5;

fprintf('Running Adam optimizer...\n');
tic;

[a_opt, history] = adam_optimizer(grad_f, a0, max_iter, alpha, beta1, beta2, epsilon, tolerance);

elapsed = toc; %end tic

%%Clamp final result to bounds
a_opt = max(a_opt, prob.a_min);
a_opt = min(a_opt, prob.a_max);

%% Evaluate final design
[loss_final, info_final] = truss_forward(a_opt, prob);

fprintf('\n── RESULTS ──────────────────────────────────────\n\n');
fprintf('  Time elapsed:  %.2f seconds\n', elapsed);
fprintf('  Iterations:    %d\n\n', size(history.w, 2));

fprintf('  %-6s %12s %12s %12s\n', 'Bar', 'Initial', 'Optimised', 'Stress');
fprintf('  %s\n', repmat('-', 1, 44));
for i = 1:prob.n_elements
    flag = '';
    if abs(info_final.stress(i)) > prob.sigma_max * 0.95
        flag = ' *';
    end
    fprintf('  %-6d %12.3f %12.3f %12.3f%s\n', ...
        i, a0(i), a_opt(i), info_final.stress(i), flag);
end

fprintf('\n  * = stress within 5%% of limit (%.1f ksi)\n\n', prob.sigma_max);
fprintf('  Initial weight:   %10.2f lbs\n', info0.weight);
fprintf('  Optimised weight: %10.2f lbs\n', info_final.weight);
fprintf('  Weight reduction: %10.1f%%\n',   ...
    100*(1 - info_final.weight / info0.weight));

