%[X1,X2]=meshgrid(linspace(-2,2),linspace(-2,2));
%[X1,X2]=meshgrid(linspace(-4,4),linspace(-4,4));
[X1, X2] = meshgrid(-2:0.1:2, -3:0.1:3);
F = (1 - X1).^2 + 100 * (X2 - X1.^2).^2; 

figure()
hold on
contour(X1,X2,F,50)
plot(x_all(1,:),x_all(2,:),'o-r')
hold off

set(gca,'FontSize',18);
xlabel('x1');
ylabel('x2');

figure()
plot(f_all);
%plot(norm(x_all-xk, 'columns'))
set(gca,'FontSize',18);
xlabel('k');
ylabel('convergence');