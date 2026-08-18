function prob = setup_truss_10bar()
% SETUP_TRUSS_10BAR  Define the classic 10-bar truss benchmark problem.
%
%   Standard test case from structural optimization literature.
%   6 nodes, 10 bars, 2 bays.  Nodes 5 & 6 are pinned supports.
%   Downward loads of 100 kips at nodes 2 and 4.
%
%       5 ──────── 3 ──────── 1
%       │╲        │╲        │
%       │  ╲      │  ╲      │
%       │    ╲    │    ╲    │
%       │      ╲  │      ╲  │
%       6 ──────── 4 ──────── 2
%                  ↓100 kips  ↓100 kips
%
%   Design variables: cross-sectional area of each bar (10 values)
%   Objective: minimise total weight
%   Constraint: element stresses must not exceed ±25 ksi
%
% OUTPUT:
%   prob  —  struct containing everything the forward model and
%            adjoint need (geometry, material, loads, precomputed
%            element stiffness and stress matrices)

    % ── Node coordinates [x, y] in inches ─────────────────────────
    prob.nodes = [
        720, 360;   % Node 1  (top-right)
        720,   0;   % Node 2  (bottom-right)
        360, 360;   % Node 3  (top-middle)
        360,   0;   % Node 4  (bottom-middle)
          0, 360;   % Node 5  (top-left, support)
          0,   0;   % Node 6  (bottom-left, support)
    ];

    % ── Element connectivity [node_i, node_j] ─────────────────────
    prob.elements = [
        3, 5;    % Bar 1:  top horizontal, left bay
        1, 3;    % Bar 2:  top horizontal, right bay
        4, 6;    % Bar 3:  bottom horizontal, left bay
        2, 4;    % Bar 4:  bottom horizontal, right bay
        3, 4;    % Bar 5:  vertical, left bay
        1, 2;    % Bar 6:  vertical, right bay
        5, 4;    % Bar 7:  diagonal left bay  (top-left → bottom-middle)
        6, 3;    % Bar 8:  diagonal left bay  (bottom-left → top-middle)
        3, 2;    % Bar 9:  diagonal right bay (top-middle → bottom-right)
        4, 1;    % Bar 10: diagonal right bay (bottom-middle → top-right)
    ];

    prob.n_nodes    = size(prob.nodes, 1);       % 6
    prob.n_elements = size(prob.elements, 1);    % 10
    prob.n_dof      = 2 * prob.n_nodes;          % 12 (2 DOFs per node)

    % ── Material properties ────────────────────────────────────────
    prob.E   = 1e4;    % Young's modulus  (ksi)
    prob.rho = 0.1;    % density          (lb / in³)

    % ── Boundary conditions ────────────────────────────────────────
    % Nodes 5 and 6 are pinned — all 4 DOFs fixed
    fixed_nodes     = [5, 6];
    prob.fixed_dofs = sort([2*fixed_nodes - 1, 2*fixed_nodes]);
    prob.free_dofs  = setdiff(1:prob.n_dof, prob.fixed_dofs);

    % ── Applied forces (kips) ──────────────────────────────────────
    prob.f        = zeros(prob.n_dof, 1);
    prob.f(2*2)   = -100;    % Node 2, y-direction
    prob.f(2*4)   = -100;    % Node 4, y-direction

    % ── Design bounds and constraints ──────────────────────────────
    prob.sigma_max = 25;      % allowable stress  (ksi)
    prob.a_min     = 0.1;     % minimum bar area   (in²)
    prob.a_max     = 35.0;    % maximum bar area   (in²)
    prob.mu        = 1e4;     % penalty weight for stress violations

    % ── Precompute element-level data ──────────────────────────────
    %    These never change during optimisation (geometry is fixed).
    %    ke0{e}     : 4×4 stiffness matrix per unit area
    %    Bstress{e} : 1×4 stress-displacement vector
    %    L(e)       : element length

    prob.L         = zeros(prob.n_elements, 1);
    prob.elem_dofs = zeros(prob.n_elements, 4);
    prob.ke0       = cell(prob.n_elements, 1);
    prob.Bstress   = cell(prob.n_elements, 1);

    for e = 1:prob.n_elements
        ni = prob.elements(e, 1);
        nj = prob.elements(e, 2);

        dx = prob.nodes(nj, 1) - prob.nodes(ni, 1);
        dy = prob.nodes(nj, 2) - prob.nodes(ni, 2);
        Le = sqrt(dx^2 + dy^2);
        c  = dx / Le;                     % cos(theta)
        s  = dy / Le;                     % sin(theta)

        prob.L(e)         = Le;
        prob.elem_dofs(e,:) = [2*ni-1, 2*ni, 2*nj-1, 2*nj];

        % Unit element stiffness (stiffness per unit area)
        %   Full stiffness = a(e) * ke0{e}
        c2 = c^2;  s2 = s^2;  cs = c*s;
        prob.ke0{e} = (prob.E / Le) * ...
            [ c2   cs  -c2  -cs;
              cs   s2  -cs  -s2;
             -c2  -cs   c2   cs;
             -cs  -s2   cs   s2];

        % Stress-displacement vector:  sigma_e = Bstress{e} * u_e
        prob.Bstress{e} = (prob.E / Le) * [-c, -s, c, s];
    end
end
