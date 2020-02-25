%% Sphere - Eigenvalue problem                       
%  script to accelerate the optimization of a quadratic potential defined
%  in R^d, with solution constrained to be on the sphere. this corresponds 
%  to solving an eigenvalue problem.

%  source for this problem:
%  https://www.manopt.org/tutorial.html#firstexample

%  REQUIRES MANOPT 5.0 toolbox https://www.manopt.org/ (add to path)

%% Clearing
clear all
close all
clc
digits(64)

%% Eigenvalue problem definion
d = 5000; % problem dimension
n = 1.05*d; %number of datapoints, set to be equal d
X = randn(n,d); %random matrix
H = (1/n) * (X'*X); %defines a very ill-conditioned ellipsoid in R^d
[eig_directions,~]=eig(H);
x_sol = eig_directions(:,end);
x_sol = x_sol/norm(x_sol);
mu = min(eig(H));
L = max(eig(H));
kappa = mu/L;
f_star = -0.5*x_sol'*H*x_sol;

%% Additional parameters
K = 300; % number or iterations
h = 1/L; % stepsize

%% Geometry-related functions 

Inner = @(x,y) InnerSphere(x,y);
Dist = @(x,y) DistSphere(x,y);
Retr = @(x) RetrSphere(x);
Exp = @(x,u) ExpSphere(x,u);
Proj = @(x,u) ProjSphere(x,u);
Log = @(x,y) LogSphere(x,y);
Gamma = @(x,y,u) GammaSphere(x,y,u); 

% try % manopt alternative (unstable)
%     M = spherefactory(d);
% catch
%     disp('Manopt toolbox missing!\n download it from https://www.manopt.org/ \n and add the folder to the path (addpath)');
% end
% Dist = @(X,Y) M.dist(X, Y);
% Proj = @(X,H) M.proj(X, H);
% Log = @(X,Y) M.log(X, Y);
% Exp = @(X,U) M.exp(X, U, 1); %no retraction after exp ->instability!
% Retr = @(X) M.retr(X, 0*X, 0);
% Gamma = @(X,Y,U) M.transp(X, Y, U);

%% Cost-related functions
f = @(x) -0.5*x'*H*x; % minus sign because we want to find the maximum eig.
egrad = @(x) -H*x;
grad = @(x) Proj(x,egrad(x)); %riemaniann gradient

%% Initialization
x0 = Retr(10*randn(d,1)); %x0 at random on the sphere
while Dist(x0,x_sol)>(pi/2)*1
    x0 = Retr(10*randn(d,1)); %x0 at random on the sphere
    disp(['Initialized at a d(x_0,x_sol) = ',num2str(Dist(x0,x_sol)),', initializing again.']);
end

%to save time series
x1 = zeros(d,K);
x2 = zeros(d,K);
y2 = zeros(d,K);   
v2 = zeros(d,K);
x3 = zeros(d,K); 
v3 = zeros(d,K);
beta2 = zeros(1,K); 
beta3 = zeros(1,K); 

%initialization to x0 random
x1(:,1) = x0;
x2(:,1) = x0;
x3(:,1) = x0;
v2(:,1) = x0;
v3(:,1) = x0;

f_val1= nan(K,1);
f_val1(1) = f(x1(:,1));
f_val2= nan(K,1);
f_val2(1) = f(x2(:,1));
f_val3= nan(K,1);
f_val3(1) = f(x3(:,1));

a = zeros(1,K);   
A = zeros(1,K);   


%% RGD
disp('Running RGD...')
for k = 1:(K-1)
    x1(:,k+1) = Exp(x1(:,k),-h*grad(x1(:,k)));
    f_val1(k+1) = f(x1(:,k+1));
end 

%% SIRNAG strongly convex
% disp('Running SIRNAG (strongly convex)...')
% for k = 1:(K-1)
%     beta_k = 0.8;
%     a_k = beta_k * v2(:,k) - sqrt(h)*grad(Exp(x2(:,k),sqrt(h)*beta_k * v2(:,k)));  
%     x2(:,k+1) = Exp(x2(:,k),sqrt(h)*a_k);
%     v2(:,k+1) = Gamma(x2(:,k),x2(:,k+1),a_k);
%     f_val2(k+1) = f(x2(:,k+1));
% end 


%% SIRNAG convex
disp('Running SIRNAG (convex)...')
for k = 1:(K-1)
    beta_k = (k-1)/(k+2);
    a_k = beta_k * v3(:,k) - sqrt(h)*grad(Exp(x3(:,k),0*h*beta_k * v3(:,k)));  
    x3(:,k+1) = Exp(x3(:,k),sqrt(h)*a_k);
    v3(:,k+1) = Gamma(x3(:,k),x3(:,k+1),a_k);
    f_val3(k+1) = f(x3(:,k+1));
end 

%% A posteriori computation of diameter and delta
disp('Computing Diameter...')
all_sampled_points = [x3];
D = 0;
for i = 1:size(all_sampled_points,2)
    for j = 1:size(all_sampled_points,2)
        D = max(D,Dist(all_sampled_points(:,i),all_sampled_points(:,j)));
    end
end

%% Getting better approx of solution
%f_star = min([f_star;f_val1;f_val3]);

%% Plotting the dynamics (iteration)
figure
h1=loglog(f_val1-f_star,'-','Linewidth',3,'Color',[0.02 0.306 0.867]);hold on
%h2=loglog(f_val2-f_star,'-','Linewidth',3,'Color','m');hold on
h3=loglog(f_val3-f_star,'-','Linewidth',3,'Color',[0.937 0.526 0.212]);hold on
h4=loglog((2*D^2)./(h^2*(1:K).^2),'--','Linewidth',2,'Color','k');hold on
h5=loglog(1:K,(0.3*L*Dist(x0,x_sol)^2)./(1:K),'-.','Linewidth',2,'Color','k');hold on
%loglog(3:K,(L*Dist(x0,x_sol)^2)./((3:K).^2),'-.','Linewidth',1,'Color','k');hold on
xlabel('$k$','Fontsize',20,'Interpreter','latex');
xlim([0,K])
ylabel('$f(x_k)-f(x^*)$','Fontsize',20,'Interpreter','latex');
l=legend([h1,h3,h4,h5],'RGD','SIRNAG (convex, Option I)','$O(1/k^2)$ bound (Thm. 5)','$O(1/k)$');
l.FontSize = 20;
set(l, 'Interpreter', 'latex','Location','best');
set(gcf,'position',[100,100,700,500]);
title('Maximum eigenvalue problem (positively curved)','Interpreter', 'latex','Fontsize',18);



%% Plotting the state
if d==3 
    figure
    theta=linspace(0,2*pi,40);
    phi=linspace(0,pi,40);
    [theta,phi]=meshgrid(theta,phi);
    rho=1;
    x=rho*sin(phi).*cos(theta);
    y=rho*sin(phi).*sin(theta);
    z=rho*cos(phi);
    surf(x,y,z);hold on
    alpha 0.2
    colormap autumn
    shading interp
    plot3(x1(1,1),x1(2,1),x1(3,1),'-k','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078]);hold on
    h1=plot3(x1(1,:),x1(2,:),x1(3,:),'-o','Linewidth',2,'Markersize',5,'Color','blue','MarkerFaceColor','blue');hold on;
    h2=plot3(x3(1,:),x3(2,:),x3(3,:),'-o','Linewidth',2,'Markersize',5,'Color','green','MarkerFaceColor','green');
    %plot3(y3(1,1:(end-1)),y3(2,1:(end-1)),y3(3,1:(end-1)),'-o','Linewidth',2,'Markersize',5,'Color','green','MarkerFaceColor','green');
    %plot3(v3(1,:),v3(2,:),v3(3,:),'-o','Linewidth',2,'Markersize',5,'Color','green','MarkerFaceColor','green');
    plot3(x_sol(1),x_sol(2),x_sol(3),'-kh','MarkerSize', 20,'MarkerFaceColor',[0.839 0.78 0.078])
    axis equal
end



%% Geometry


function r = RetrSphere(x)
	r = x/norm(x);
end

function r = ExpSphere(x,u)
    if norm(u)~=0
        r = cos(norm(u))*x + sin(norm(u))*(u/norm(u)); %exp map
    else
        r = x;
    end
    r = RetrSphere(r); %numerical instability otherwise: exists M! to explore...
end

function d = DistSphere(x,y)
    d= acos(x'*y); %discuss what happens if x^Ty = 1
end

function p = ProjSphere(x,u)
    p=u-x*x'*u; %projection of a vector u in ambient space to TM_x
end

function r = LogSphere(x,u)
    if norm(ProjSphere(x,u-x))~=0
        r = (DistSphere(x,u)/norm(ProjSphere(x,u-x)))*ProjSphere(x,u-x); %log map
    else
        r = 0;
    end
end

function r = GammaSphere(x,y,u)
    r = ProjSphere(y,u);
end






