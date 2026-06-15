% Newton Method S. Marwitz, May 2019, May 2020

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

%% Task 3 c)
%initial guess
xk=x0;
% initial descent direction
dk = -h(x0)\g(x0);
% initialize iteration Counter
k=1;
%initialize vector to store intermediate steps needed  for Task 3 d)
x_all = [xk,];
f_all = [f(xk),];
% evaluate descent direction
    dk = -h(xk)\g(xk);
while(norm(dk) > tolerance) && (k < max_iter) %stopping criterion
    
    
    %compute next point
    xk= xk+dk;
    
    %store new point
    x_all = [x_all, xk];
    f_all = [f_all, f(xk)];
    
    %Update the descent direction
    dk = -h(xk)\g(xk);
    %increase iteration counter
    k=k+1;
end


% Print the optimal point and number of iterations
fprintf('\nThe optimal point (%f,%f)  (objective value = %f)  was found after %d iterations \n', xk(1),xk(2),f(xk), k-2);


% visualize the optimizer steps and convergence
visualize
