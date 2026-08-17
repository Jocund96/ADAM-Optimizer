% Method of Steepest Descent, T. Lahmer, S. Marwitz, May 2019, May 2020
clear all
close all

% function definition
g = @(x) [ -2*(1-x(1)) - 400*x(1)*(x(2) - x(1)^2); 200*(x(2) - x(1)^2)]; % Rosenbrock
h = @(x) [1200*x(1)^2 - 400*x(2) + 2, -400*x(1); -400*x(1), 200]; % Rosenbrock
f = @(x) (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2; % Rosenbrock
% initial guess / starting point
x0 = [-1.5; 2.0]; 
% optimal point
xopt= fminsearch(f,x0)
% print the optimal point
fprintf("\nThe optimal point (%f,%f) (objective value = %f) was found by Matlab's built in optimizer \n", xopt(1),xopt(2),f(xopt));
tolerance = 1e-5;
max_iter= 15000; 
t = 0.01;  

% initial guess
xk=x0;
% initial gradient
gk = g(x0);
% initial descent direction
dk = -g(x0);
% initialize iteration counter
k=1;
%initialize vector to store intermediate steps 
x_all = [xk,];
f_all = [f(xk),];
while (norm(gk) > tolerance) && (k < max_iter) %stopping criterion (k<64) %
 
    %evaluate optimal step size / exact line search
    f_line = @(t) f(xk + t * dk);
    tk = fminbnd(f_line, 0, 0.002); 
    
    %compute next point
    xk= xk+tk*dk;
    
    %store new point needed for Task f)
    x_all = [x_all,xk];
    f_all = [f_all, f(xk)];
    %increase iteration counter
    k=k+1;    
    
    % evaluate exact descent direction
    gk = g(xk); 
    % evaluate next descent direction
    dk = -g(xk);
end

% Print the optimal point and number of iterations
fprintf('\nThe optimal point (%f,%f)  (objective value = %f)  was found after %d iterations \n', xk(1),xk(2),f(xk), k-1);

%Task e)
% Print the optimality criteria
fprintf('\nThe norm of the gradient is  (%f) eigenvalues of the hessian are (%f,%f) \n', norm(g(xk)),eigs(h(xk)));

% Task f) Task g)
% visualize the optimizer steps and convergence

visualize


