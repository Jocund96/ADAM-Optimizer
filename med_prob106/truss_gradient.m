% TRUSS_GRADIENT  Compute the gradient for the Adam optimizer.
function g = truss_gradient(a, prob)

    % Enforce bounds — prevents singular stiffness matrix
    a = max(a, prob.a_min);
    a = min(a, prob.a_max);

    % Forward pass (assemble K, solve Ku=f, compute stresses)
    [~, info] = truss_forward(a, prob);

    % Backward pass (adjoint — compute gradient)
    g = truss_adjoint(a, prob, info);
end
