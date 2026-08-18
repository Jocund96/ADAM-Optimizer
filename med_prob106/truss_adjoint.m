% Compute the gradient dJ/da using the adjoint method.
function g = truss_adjoint(a, prob, info)

%   The three steps are:
%     1. Compute ∂J/∂u  (how the penalty changes with displacements)
%     2. Solve the adjoint equation  K λ = −∂J/∂u  (reuses LU from forward)
%     3. Assemble the gradient  dJ/da_i = ρL_i − λᵀ · ke0_i · u_i

    n_dof = prob.n_dof;
    n_el  = prob.n_elements;
    free  = prob.free_dofs;

    % ══════════════════════════════════════════════════════════
    % STEP 1:  Compute  ∂J/∂u  (from the stress penalty)
    %
    %   J = weight + penalty
    %   Weight does not depend on u, so ∂W/∂u = 0.
    %   Only the penalty contributes:
    %
    %   penalty = μ · Σ max(0, |σ_e|/σ_max − 1)²
    %   σ_e     = B_e · u_e
    %
    %   Chain rule for each element:
    %     ∂P/∂u_e = (∂P/∂σ_e) · (∂σ_e/∂u_e)
    %             = (∂P/∂σ_e) · B_e
    %
    %   where  ∂P/∂σ_e = 2μ · max(0, |σ|/σ_max − 1) · sign(σ) / σ_max
    %
    %   We scatter each element's contribution to the global ∂J/∂u.
    % ══════════════════════════════════════════════════════════
    dJ_du = zeros(n_dof, 1);

    for e = 1:n_el
        sigma_e   = info.stress(e);
        violation = abs(sigma_e) / prob.sigma_max - 1;

        if violation > 0
            % Derivative of penalty w.r.t. this element's stress
            dP_dsigma = 2 * prob.mu * violation * sign(sigma_e) / prob.sigma_max;

            % Scatter to global DOFs:  ∂P/∂u += dP_dsigma · B_eᵀ
            dofs = prob.elem_dofs(e, :);
            dJ_du(dofs) = dJ_du(dofs) + dP_dsigma * prob.Bstress{e}';
        end
    end

    % ══════════════════════════════════════════════════════════
    % STEP 2:  Solve the adjoint equation
    %
    %          K · λ = −∂J/∂u
    %
    %   K is symmetric, so Kᵀ = K.  We reuse the LU factors
    %   from the forward solve — NO new factorisation needed.
    %   This is the key efficiency of the adjoint + LU approach:
    %   one factorisation serves both the forward and backward pass.
    % ══════════════════════════════════════════════════════════
    rhs         = -dJ_du(free);
    lambda_free = info.U_lu \ (info.L_lu \ (info.P_lu * rhs));

    lambda       = zeros(n_dof, 1);
    lambda(free) = lambda_free;

    % ══════════════════════════════════════════════════════════
    % STEP 3:  Assemble the gradient, element by element
    %
    %   dJ/da_i  =  (∂J/∂a_i)|direct  +  λᵀ · (−∂K/∂a_i · u)
    %
    %   Direct term (from weight):   ρ · L_i
    %
    %   Adjoint term:  Since K = Σ a_i · ke0_i,
    %                  ∂K/∂a_i = ke0_i  (the unit stiffness)
    %
    %   So:  dJ/da_i = ρ·L_i  −  λ_eᵀ · ke0_i · u_e
    %
    %   Each element contributes independently — we loop once.
    % ══════════════════════════════════════════════════════════
    g = zeros(n_el, 1);

    for e = 1:n_el
        dofs  = prob.elem_dofs(e, :);
        u_e   = info.u(dofs);
        lam_e = lambda(dofs);

        % Direct: weight gradient
        g(e) = prob.rho * prob.L(e);

        % Adjoint: sensitivity through displacements
        g(e) = g(e) - lam_e' * prob.ke0{e} * u_e;
    end
end
