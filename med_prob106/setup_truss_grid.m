function prob = setup_truss_grid(nx, ny, bay_width, bay_height)
% SETUP_TRUSS_GRID  Generate a rectangular grid truss with ~100+ bars.
%
%   Instead of hand-defining 10 bars, this function builds a truss
%   automatically from a grid.  You control the size:
%
%     nx = number of bays in x-direction
%     ny = number of bays in y-direction
%
%   Each bay gets:  2 horizontal + 2 vertical + 2 diagonal bars
%   Total bars  ≈  4·nx·ny + nx + ny
%
%   ┌──────┬──────┬──────┬──────┐
%   │╲    ╱│╲    ╱│╲    ╱│╲    ╱│    ny = 2, nx = 4
%   │  ╲╱  │  ╲╱  │  ╲╱  │  ╲╱  │    → 42 bars, 15 nodes
%   ├──────┼──────┼──────┼──────┤
%   │╲    ╱│╲    ╱│╲    ╱│╲    ╱│
%   │  ╲╱  │  ╲╱  │  ╲╱  │  ╲╱  │
%   └──────┴──────┴──────┴──────┘
%   ▲                            ▲
%   fixed                       fixed
%
%   QUICK REFERENCE — pick nx, ny to get your target bar count:
%
%     nx=5, ny=4  →  89 bars,  30 nodes
%     nx=6, ny=4  → 106 bars,  35 nodes
%     nx=7, ny=3  →  94 bars,  32 nodes
%     nx=8, ny=3  → 107 bars,  36 nodes
%     nx=7, ny=4  → 123 bars,  40 nodes
%
% INPUTS:
%   nx         — bays in x     (default: 6)
%   ny         — bays in y     (default: 4)
%   bay_width  — width per bay in inches  (default: 360)
%   bay_height — height per bay in inches (default: 360)
%
% OUTPUT:
%   prob — same struct format as setup_truss_10bar, compatible
%          with truss_forward, truss_adjoint, truss_gradient

    % ── Defaults ───────────────────────────────────────────────
    if nargin < 1, nx = 6;   end
    if nargin < 2, ny = 4;   end
    if nargin < 3, bay_width  = 360; end
    if nargin < 4, bay_height = 360; end

    % ══════════════════════════════════════════════════════════
    % STEP 1:  Generate nodes on a rectangular grid
    %
    %   (nx+1) columns × (ny+1) rows of nodes
    %   Numbering: left-to-right, bottom-to-top
    %
    %   Node index for grid position (ix, iy):
    %     node_id = iy * (nx+1) + ix + 1
    % ══════════════════════════════════════════════════════════
    n_cols = nx + 1;
    n_rows = ny + 1;
    n_nodes = n_cols * n_rows;

    nodes = zeros(n_nodes, 2);
    for iy = 0:ny
        for ix = 0:nx
            idx = iy * n_cols + ix + 1;
            nodes(idx, :) = [ix * bay_width, iy * bay_height];
        end
    end

    % ══════════════════════════════════════════════════════════
    % STEP 2:  Generate element connectivity
    %
    %   For each bay, create:
    %     - horizontal bars (bottom and top of bay)
    %     - vertical bars   (left and right of bay)
    %     - diagonal bars   (both ╲ and ╱)
    %
    %   Duplicates are avoided by only adding bottom/left/diag
    %   per bay, then sweeping up top-row and right-column edges.
    % ══════════════════════════════════════════════════════════
    elements = [];

    % Helper: grid position → node index
    nid = @(ix, iy) iy * n_cols + ix + 1;

    % Horizontal bars
    for iy = 0:ny
        for ix = 0:(nx-1)
            elements = [elements; nid(ix,iy), nid(ix+1,iy)];
        end
    end

    % Vertical bars
    for ix = 0:nx
        for iy = 0:(ny-1)
            elements = [elements; nid(ix,iy), nid(ix,iy+1)];
        end
    end

    % Diagonal bars (both directions in each bay)
    for iy = 0:(ny-1)
        for ix = 0:(nx-1)
            % ╱ diagonal: bottom-right to top-left
            elements = [elements; nid(ix+1,iy), nid(ix,iy+1)];
            % ╲ diagonal: bottom-left to top-right
            elements = [elements; nid(ix,iy), nid(ix+1,iy+1)];
        end
    end

    n_elements = size(elements, 1);

    % ══════════════════════════════════════════════════════════
    % STEP 3:  Boundary conditions
    %
    %   Pin the bottom-left and bottom-right corner nodes.
    %   (same as the 10-bar: supports on the left edge)
    %
    %   For a cantilever (fixed left, loaded right):
    %     fix all nodes in the leftmost column (ix = 0)
    % ══════════════════════════════════════════════════════════
    prob.n_nodes    = n_nodes;
    prob.n_elements = n_elements;
    prob.n_dof      = 2 * n_nodes;
    prob.nodes      = nodes;
    prob.elements   = elements;

    % Fix all nodes on the left edge (ix = 0)
    fixed_nodes = [];
    for iy = 0:ny
        fixed_nodes = [fixed_nodes, nid(0, iy)];
    end
    prob.fixed_dofs = sort([2*fixed_nodes - 1, 2*fixed_nodes]);
    prob.free_dofs  = setdiff(1:prob.n_dof, prob.fixed_dofs);

    % ══════════════════════════════════════════════════════════
    % STEP 4:  Applied forces
    %
    %   Downward load on all nodes along the right edge
    %   and a downward load on the bottom-right corner
    % ══════════════════════════════════════════════════════════
    prob.f = zeros(prob.n_dof, 1);

    % Load on bottom-right corner: large point load
    corner_node = nid(nx, 0);
    prob.f(2 * corner_node) = -200;   % 200 kips down

    % Distributed load along top edge (gravity-like)
    for ix = 1:nx
        top_node = nid(ix, ny);
        prob.f(2 * top_node) = -50;   % 50 kips each
    end

    % ══════════════════════════════════════════════════════════
    % STEP 5:  Material and constraints (same as 10-bar)
    % ══════════════════════════════════════════════════════════
    prob.E         = 1e4;     % Young's modulus (ksi)
    prob.rho       = 0.1;     % density (lb/in³)
    prob.sigma_max = 25;      % allowable stress (ksi)
    prob.a_min     = 0.1;     % minimum area (in²)
    prob.a_max     = 35.0;    % maximum area (in²)
    prob.mu        = 1e4;     % penalty weight

    % ══════════════════════════════════════════════════════════
    % STEP 6:  Precompute element data (same as 10-bar)
    %
    %   This loop is identical to setup_truss_10bar.
    %   ke0{e} and Bstress{e} are used by both the forward
    %   model and the adjoint — they never change during
    %   optimisation because geometry is fixed.
    % ══════════════════════════════════════════════════════════
    prob.L         = zeros(n_elements, 1);
    prob.elem_dofs = zeros(n_elements, 4);
    prob.ke0       = cell(n_elements, 1);
    prob.Bstress   = cell(n_elements, 1);

    for e = 1:n_elements
        ni = elements(e, 1);
        nj = elements(e, 2);

        dx = nodes(nj, 1) - nodes(ni, 1);
        dy = nodes(nj, 2) - nodes(ni, 2);
        Le = sqrt(dx^2 + dy^2);
        c  = dx / Le;
        s  = dy / Le;

        prob.L(e) = Le;
        prob.elem_dofs(e, :) = [2*ni-1, 2*ni, 2*nj-1, 2*nj];

        c2 = c^2;  s2 = s^2;  cs = c*s;
        prob.ke0{e} = (prob.E / Le) * ...
            [ c2   cs  -c2  -cs;
              cs   s2  -cs  -s2;
             -c2  -cs   c2   cs;
             -cs  -s2   cs   s2];

        prob.Bstress{e} = (prob.E / Le) * [-c, -s, c, s];
    end

    % ── Print summary ──────────────────────────────────────────
    fprintf('Grid truss created:  %d × %d bays\n', nx, ny);
    fprintf('  Nodes:            %d\n', n_nodes);
    fprintf('  Bars (variables): %d\n', n_elements);
    fprintf('  Total DOFs:       %d\n', prob.n_dof);
    fprintf('  Free DOFs:        %d\n', length(prob.free_dofs));
    fprintf('  Fixed nodes:      %d  (left edge)\n', length(fixed_nodes));
    fprintf('\n');
end
