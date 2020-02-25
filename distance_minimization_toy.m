%% Toy example: minimization of the geodesic distance
% we minimize 0.5 d(x,p)^2 with respoect to x, where d is the riemannian 
% distance and p a point on the manifold


%claning
clear all
close all
beep off
clc

%% Parameters 
h = 0.1; % integration stepsize
K = 1e4; % number or iterations
d = 3; % problem dimension
R = 1; %radius of hyperbolic plane
D = 1; %distance to the minimizer

%% Definition of geometry
Inner = @(x,y) InnerHy(x,y);
Dist = @(x,y) DistHy(x,y,R);
Retr = @(x) RetrHy(x,R);
Exp = @(x,u) ExpHy(x,u,R);
Proj = @(x,u) ProjHy(x,u,R);
Log = @(x,y) LogHy(x,y,R);
Gamma = @(x,y,u) GammaHy(x,y,u,R);
    
%% Geometry check
disp('Testing the implementation of geometry...')
x1=Retr(10*randn(3,1));x0=Retr(10*randn(3,1));
disp(['||x0||^2_M + R^2 = ', num2str(Inner(x0,x0)+R^2)])
disp(['||x1||^2_M + R^2 = ', num2str(Inner(x1,x1)+R^2)])
disp(['||Exp(x0, Log(x0,x1))-x1|| = ', num2str(norm(Exp(x0, Log(x0,x1))-x1))])
disp(['||Log(x0,x1)||_M - Dist(x0,x1) = ', num2str(sqrt(Inner(Log(x0,x1),Log(x0,x1)))-Dist(x0,x1))])
disp(['||Log(x1,x0)||_M - Dist(x0,x1) = ', num2str(sqrt(Inner(Log(x1,x0),Log(x1,x0)))-Dist(x0,x1))])
disp(['||Proj(x0,Log(x0,x1))-Log(x0,x1)|| = ', num2str(norm(Proj(x0,Log(x0,x1))-Log(x0,x1)))])
v = Proj(x0,randn(3,1));
disp(['||Proj(x0,v)-v|| = ', num2str(norm(Proj(x0,v)-v))])
disp('-- All of the above must be very close to zero!! --')
disp(' ')

%% Function Definition
x_star = zeros(d,1);
x_star(end)=R;

f = @(x) f_distance_squared(x,x_star,R);
grad = @(x) grad_distance_squared(x,x_star,R);

% f = @(x) f_one_over_x(x,R);
% grad = @(x) grad_one_over_x(x,R);

%% Initialization
displacement = sqrt(D^2/(d-1))*ones(d,1);
displacement(end)=0;
x0 = Exp(x_star,displacement); %starting from Dist=D to the minimizer

x1 = zeros(d,K);
x2 = zeros(d,K);
v2 = zeros(d,K);
x3 = zeros(d,K);
v3 = zeros(d,K);
x4 = zeros(d,K);
v4 = zeros(d,K);
x5 = zeros(d,K);
v5 = zeros(d,K);

x1(:,1) = Retr(x0);
x2(:,1) = Retr(x0);
x3(:,1) = Retr(x0);
x4(:,1) = Retr(x0);
x5(:,1) = Retr(x0);
f_val1= zeros(K,1);
f_val2= zeros(K,1);  
f_val3= zeros(K,1);
f_val4= zeros(K,1);  
f_val5= zeros(K,1);  
f_val1(1) = f(x1(:,1));
f_val2(1) = f(x2(:,1));
f_val4(1) = f(x4(:,1));
f_val5(1) = f(x5(:,1));

%% Computing zeta
D = Dist(x1(:,1),x_star);
curvature = 1/R;
zeta = sqrt(curvature)*D*coth(sqrt(curvature)*D);

%% Solving the problem
disp('Running the algorithm...')

for k = 1:(K-1)
    %GD
    x1(:,k+1) = Exp(x1(:,k),-h^2*grad(x1(:,k)));
    f_val1(k+1) = f(x1(:,k+1));
    
    %SIRNAG (no extrapolation)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v2(:,k) - h*grad(x2(:,k));
    x2(:,k+1) = Exp(x2(:,k),h*a_k);
    v2(:,k+1) = Gamma(x2(:,k),x2(:,k+1),a_k);
    f_val2(k+1) = f(x2(:,k+1)); 
    if 0 %for debugging
        disp(' ')
        disp(['Interation ', num2str(k)])
        disp(['||grad(x2(k))|| = ',num2str(norm(h*grad(x2(:,k))))]);
        disp(['||a_k|| = ',num2str(norm(a_k))]);    
        disp(['||v2(k+1)|| = ',num2str(norm(v2(:,k+1)))]); 
        disp(['||a_k||_M = ',num2str(Inner(a_k,a_k))]);    
        disp(['||v2(k+1)||_M = ',num2str(Inner(v2(:,k+1),v2(:,k+1)))]); 
        test = Exp(x2(:,k+1),v2(:,k+1));
        test2 = Exp(x2(:,k+1),a_k);
        disp(['||exp(x2(k+1),a_k)||_M +R^2 = ',num2str(Inner(test2,test2)+R^2)]); 
        disp(['||exp(x2(k+1),v2(k+1))||_M +R^2 = ',num2str(Inner(test,test)+R^2)]); 
        disp(['||x2(k+1)||_M + R^2 = ',num2str(Inner(x2(:,k+1),x2(:,k+1))+R^2),', ||x2(k+1)|| = ',num2str(norm(x2(:,k+1)))]);
    end

    %SIRNAG (extrapolation)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v3(:,k) - h*grad(Exp(x3(:,k),h*beta_k * v3(:,k)));  
    x3(:,k+1) = Exp(x3(:,k),h*a_k);
    v3(:,k+1) = Gamma(x3(:,k),x3(:,k+1),a_k);
    f_val3(k+1) = f(x3(:,k+1)); 
    
    %SIRNAG (strongly convex, no extrapolation)
    beta_k = 1-h*(sqrt(zeta)+1/sqrt(zeta));
    a_k = beta_k * v4(:,k) - h*grad(x4(:,k));  
    x4(:,k+1) = Exp(x4(:,k),h*a_k);
    v4(:,k+1) = Gamma(x4(:,k),x4(:,k+1),a_k);
    f_val4(k+1) = f(x4(:,k+1)); 
    
    %SIRNAG (strongly convex, no extrapolation)
    beta_k = 1-h*(sqrt(zeta)+1/sqrt(zeta));
    a_k = beta_k * v5(:,k) - h*grad(Exp(x5(:,k),h*beta_k * v5(:,k)));  
    x5(:,k+1) = Exp(x5(:,k),h*a_k);
    v5(:,k+1) = Gamma(x5(:,k),x5(:,k+1),a_k);
    f_val5(k+1) = f(x5(:,k+1)); 

end 

%% Computing the diameter
D1 = Dist(x1(:,1),x1(:,end));
D2 = Dist(x2(:,1),x2(:,end));
D3 = Dist(x3(:,1),x3(:,end));


%% Plotting the dynamics
h2=loglog(f_val2,'-','Linewidth',3,'Color',[0.066 0.661 0.757]);hold on
h3=loglog(f_val3,'-','Linewidth',3,'Color',[0.937 0.526 0.212]);hold on
h4=loglog(f_val4,'-','Linewidth',3,'Color',[0.111 0.8 0.147]);hold on
h5=loglog(f_val5,'-','Linewidth',3,'Color','m');hold on
h1=loglog(f_val1,'-','Linewidth',3,'Color',[0.02 0.306 0.867]);hold on
%loglog(4*f_val1(1)./(1:K),'--','Linewidth',2,'Color','k');hold on
h6=loglog((2*zeta*D2^2)./(h^2*(1:K).^2),'--','Linewidth',2,'Color','k');hold on

loglog(20:K,10000./((20:K).^(3.6)),'-.','Linewidth',1,'Color','k');hold on
xlabel('$k$','Fontsize',18,'Interpreter','latex')
ylabel('$f(x_k)-f(x^*)$','Fontsize',18,'Interpreter','latex')
l=legend([h1,h2,h3,h4,h5,h6],{'RGD','SIRNAG (geod. convex, Option I)','SIRNAG (geod. convex, Option II)','SIRNAG (geod. strongly convex, Option I)','SIRNAG (geod. strongly convex, Option II)','$\mathcal{O}(1/k^2)$ bound (Thm. 5)'});
l.FontSize = 14;
set(l, 'Interpreter', 'latex','Location','best')
title('$f(x) = \frac{1}{2}d(x,p)^2$ with $p\in M$ (negatively curved)','Interpreter', 'latex','Fontsize',18);
ylim([1e-13,1e5])
text(2e3,5e-8,'$\mathcal{O}(1/k^{3.6})$','Fontsize',15,'Interpreter', 'latex')
set(gcf,'position',[100,100,700,500])

%% Plotting the manifold
if 0
    if d==3 
        figure
        x=linspace(-30,30,100);
        y=linspace(-30,30,100);
        [x,y]=meshgrid(x,y);
        z=sqrt(R^2+x.^2+y.^2);
        surf(x,y,z);hold on
        alpha 0.2
        colormap autumn
        shading interp
        plot3(x_star(1),x_star(2),x_star(3),'-kh','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078]);hold on
        plot3(x1(1,1),x1(2,1),x1(3,1),'-k','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078]);hold on
        h1=plot3(x1(1,:),x1(2,:),x1(3,:),'-o','Linewidth',2,'Markersize',5,'Color','blue','MarkerFaceColor','blue');hold on;
        h2=plot3(x2(1,1:5),x2(2,1:5),x2(3,1:5),'-o','Linewidth',2,'Markersize',5,'Color','magenta','MarkerFaceColor','magenta');
    end
end


%% Additional functions 
function y = f_distance_squared(x,x_star,R)
    y = 0.5*DistHy(x,x_star,R)^2;
end

function g = grad_distance_squared(x,x_star,R)
    g = -LogHy(x,x_star,R);
end

function y = f_one_over_x(x,R)
    y = 1/x(end);
end

function g = grad_one_over_x(x,R)
    g = 0*x;
    g(end) = 1/(x(end)^2);
    g = ProjHy(x,g,R);
end


% Hyperbolic geometry

function p = InnerHy(x,y) 
    p= x(1:end-1)'*y(1:end-1) - x(end)'*y(end);
end

function d = DistHy(x,y,R)
    if -InnerHy(x,y)>R^2
        d = R*acosh(-InnerHy(x,y)/R^2);
    else
        d=0;
    end

end

function y = RetrHy(x,R)
	y = x;
    y(end) = sqrt(R^2+sum(y(1:(end-1)).^2));
end

function y = ExpHy(x,u,R)
    if norm(u)<eps
        y=x;
    else
        u = ProjHy(x,u,R);
        normu = sqrt(InnerHy(u,u));
        y=x*cosh(normu/R) + sinh(normu/R)*(R*u)./normu;
        y = RetrHy(y,R);
    end
end

function p = ProjHy(x,u,R)
    p= u + (InnerHy(x,u)/(R^2))*x;
end

function u = LogHy(x,y,R)
    if sqrt(InnerHy(x,y)^2-R^4)<eps
        u = 0*R*x;
    else
        u = (acosh(-InnerHy(x,y)/R^2)/sqrt(InnerHy(x,y)^2/R^4-1))*(y + InnerHy(x,y)/R^2*x);
        u = ProjHy(x,u,R);
    end
end

function r = GammaHy(x,y,u,R)
    if DistHy(x,y,R)>eps
        r = u - (InnerHy(LogHy(x,y,R),u)/DistHy(x,y,R))*(LogHy(x,y,R)+LogHy(y,x,R));
        r = ProjHy(y,r,R);
    else
        r = u;
        r = ProjHy(y,r,R);
    end
end






