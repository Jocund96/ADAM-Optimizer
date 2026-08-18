function [loss, info] = truss_forward(a, prob)
% TRUSS_FORWARD  Run the forward finite-element solve for the truss.
%
%   Given bar areas a, this function:
%     1. Assembles the global stiffness matrix  K(a)
%     2. Solves  K u = f  via LU decomposition
%     3. Computes element stresses from displacements
%     4. Computes the objective = weight + stress penalty
%
% INPUTS:
%   a     — (n_elements × 1) bar cross-sectional areas
%   prob  — problem struct from setup_truss_10bar()
%
% OUTPUTS:
%   loss  — scalar objective value (weight + penalty)
%   info  — struct with all intermediate results needed by the adjoint:
%           .u        : full displacement vector
%           .stress   : element stresses
%           .weight   : structural weight
%           .penalty  : stress violation penalty
%           .L_lu, .U_lu, .P_lu : LU factors of K (reused by adjoint)

    n_dof = prob.n_dof;
    n_el  = prob.n_elements;
    free  = prob.free_dofs;

    % ══════════════════════════════════════════════════════════
    % STEP 1:  Assemble global stiffness matrix
    %          K = Σ  a(e) · ke0{e}   (sum over elements)
    % ══════════════════════════════════════════════════════════
    K = zeros(n_dof);
    for e = 1:n_el
        dofs = prob.elem_dofs(e, :);
        K(dofs, dofs) = K(dofs, dofs) + a(e) * prob.ke0{e};
    end

    % ══════════════════════════════════════════════════════════
    % STEP 2:  Solve  K · u = f  using LU decomposition
    %
    %   Only the free DOFs are solved (supports are fixed).
    %   LU factors are stored for reuse in the adjoint solve,
    %   so we don't factorise the same matrix twice.
    % ══════════════════════════════════════════════════════════
    Kf = K(free, free);
    ff = prob.f(free);

    [L_lu, U_lu, P_lu] = lu(Kf);                 % factorise once
    u_free = U_lu \ (L_lu \ (P_lu * ff));         % solve via LU

    u       = zeros(n_dof, 1);
    u(free) = u_free;

    % ══════════════════════════════════════════════════════════
    % STEP 3:  Compute element stresses
    %
    %   For each bar:  σ_e = B_e · u_e
    %   where B_e = (E / L_e) · [-cos  -sin  cos  sin]
    % ══════════════════════════════════════════════════════════
    stress = zeros(n_el, 1);
    for e = 1:n_el
        u_e      = u(prob.elem_dofs(e, :));
        stress(e) = prob.Bstress{e} * u_e;
    end

    % ══════════════════════════════════════════════════════════
    % STEP 4:  Compute objective  J = weight + penalty
    %
    %   Weight:   W = Σ  ρ · a(e) · L(e)
    %   Penalty:  P = μ · Σ  max(0, |σ_e|/σ_max − 1)²
    %
    %   The penalty enforces stress constraints softly.
    %   When all stresses are below σ_max, penalty = 0.
    % ══════════════════════════════════════════════════════════
    weight  = sum(prob.rho .* a(:) .* prob.L);

    violation = max(0, abs(stress) ./ prob.sigma_max - 1);
    penalty   = prob.mu * sum(violation .^ 2);

    loss = weight + penalty;

    % ── Store everything the adjoint will need ─────────────────
    info.u       = u;
    info.stress  = stress;
    info.weight  = weight;
    info.penalty = penalty;
    info.L_lu    = L_lu;
    info.U_lu    = U_lu;
    info.P_lu    = P_lu;
end
