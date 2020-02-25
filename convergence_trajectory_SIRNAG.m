%% Convergence of SIRNAG                       
%  We test the convergence of SIRNAG to the flow of RNAG-ODE
%  as the stepsize goes to zero

%cleaning
clear all
close all
beep off
clc

%% Parameters 
h1 = 1e-5; % integration stepsize 1 - supersmall (ODE)
h2 = 5e-2; % integration stepsize 2 - small
h3 = 4e-1; % integration stepsize 3 - fair
h4 = 8e-1; % integration stepsize 4 - big

K4 = 25; %number of steps for the algorithm with biggest step
T = K4*h4; %corresponding time variable

K3 = T/h3; % equivalent number or iterations for stepsize 3
K2 = T/h2; % equivalent number or iterations for stepsize 2
K1 = floor(T/h1); % equivalent number or iterations for stepsize 1

%computation of the corresponding time variables
t1 = h1*(0:(K1-1));
t2 = h2*(0:(K2-1));
t3 = h3*(0:(K3-1));
t4 = h4*(0:(K4-1));

%% Definition of geometry
d = 3; % problem dimension
geometry = 'hyperbolic';
R = 1; %radius of hyperbolic plane
D = 1; %distance to the minimizer

if strcmpi(geometry, 'hyperbolic')
    Inner = @(x,y) InnerHy(x,y); %see end of the spript for definitions
    Dist = @(x,y) DistHy(x,y,R);
    Retr = @(x) RetrHy(x,R);
    Exp = @(x,u) ExpHy(x,u,R);
    Proj = @(x,u) ProjHy(x,u,R);
    Log = @(x,y) LogHy(x,y,R);
    Gamma = @(x,y,u) GammaHy(x,y,u,R);
    
elseif strcmpi(geometry, 'euclidean')
    Inner = @(x,y) InnerEuclidean(x,y);
    Dist = @(x,y) DistEuclidean(x,y);
    Retr = @(x) RetrEuclidean(x);
    Exp = @(x,u) ExpEuclidean(x,u);
    Proj = @(x,u) ProjEuclidean(x,u);
    Log = @(x,y) LogEuclidean(x,y);
    Gamma = @(x,y,u) GammaEuclidean(x,y,u);
    
elseif strcmpi(geometry, 'spherical')
    Inner = @(x,y) InnerSphere(x,y);
    Dist = @(x,y) DistSphere(x,y);
    Retr = @(x) RetrSphere(x);
    Exp = @(x,u) ExpSphere(x,u);
    Proj = @(x,u) ProjSphere(x,u);
    Log = @(x,y) LogSphere(x,y);
    Gamma = @(x,y,u) GammaSphere(x,y,u);    
    
end


%% Problem definion
x_star = zeros(d,1);
x_star(end)=R;
f = @(x) 0.5*Dist(x,x_star)^2;
grad = @(x) -Log(x,x_star);

%% Initialization
displacement = sqrt(D^2/(d-1))*ones(d,1);
displacement(end)=0;
x0 = Exp(x_star,displacement); %starting from Dist=D to the minimizer

x1 = zeros(d,K1);
v1 = zeros(d,K1);
x2 = zeros(d,K2);
v2 = zeros(d,K2);
x3 = zeros(d,K3);
v3 = zeros(d,K3);
x4 = zeros(d,K4);
v4 = zeros(d,K4);

x1(:,1) = Retr(x0);
x2(:,1) = Retr(x0);
x3(:,1) = Retr(x0);
x4(:,1) = Retr(x0);
f_val1= zeros(K1,1);
f_val2= zeros(K2,1); 
f_val3= zeros(K3,1);  
f_val4= zeros(K4,1);  
f_val1(1) = f(x1(:,1));
f_val2(1) = f(x2(:,1));
f_val3(1) = f(x3(:,1));
f_val4(1) = f(x4(:,1));


%% Computing zeta
D = Dist(x1(:,1),x_star);
curvature = 1/R;
zeta = sqrt(curvature)*D*coth(sqrt(curvature)*D);

%% Solving the problem
disp('Running numerical integration...')

%SIE-A supersmall step
for k = 1:(K1-1)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v1(:,k) - h1*grad(x1(:,k));
    x1(:,k+1) = Exp(x1(:,k),h1*a_k);
    v1(:,k+1) = Gamma(x1(:,k),x1(:,k+1),a_k);
    f_val1(k+1) = f(x1(:,k+1)); 
end    

%SIE-A small step
for k = 1:(K2-1)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v2(:,k) - h2*grad(x2(:,k));
    x2(:,k+1) = Exp(x2(:,k),h2*a_k);
    v2(:,k+1) = Gamma(x2(:,k),x2(:,k+1),a_k);
    f_val2(k+1) = f(x2(:,k+1)); 
end

%SIE-B big step
for k = 1:(K3-1)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v3(:,k) - h3*grad(x3(:,k));  
    x3(:,k+1) = Exp(x3(:,k),h3*a_k);
    v3(:,k+1) = Gamma(x3(:,k),x3(:,k+1),a_k);
    f_val3(k+1) = f(x3(:,k+1)); 
end 

%SIE-B superbig step
for k = 1:(K4-1)
    beta_k = (k-1)/(k+2*zeta);
    a_k = beta_k * v4(:,k) - h4*grad(x4(:,k));  
    x4(:,k+1) = Exp(x4(:,k),h4*a_k);
    v4(:,k+1) = Gamma(x4(:,k),x4(:,k+1),a_k);
    f_val4(k+1) = f(x4(:,k+1)); 
end 


%% Computing the discretization error
for k = 1:K4
    [~,idx3(k)] = min(abs(t4(k)-t3));
    [~,idx2(k)] = min(abs(t4(k)-t2));
    [~,idx1(k)] = min(abs(t4(k)-t1));

    error4(k) = Dist(x4(:,k),x1(:,idx1(k)));
    error3(k) = Dist(x3(:,idx3(k)),x1(:,idx1(k)));
    error2(k) = Dist(x2(:,idx2(k)),x1(:,idx1(k)));
end



%% Plotting the suboptimality
col = [0.016 0.318 0.686;0.086 0.62 0.114;0.937 0.526 0.212;0.833 0.243 0.533];
figure
subplot(1,2,1)
plot(t1(idx1),error2,'-','Linewidth',3,'Color',col(1,:));hold on
plot(t2(idx2),error3,'-','Linewidth',3,'Color',col(2,:));hold on
plot(t2(idx2),error4,'-','Linewidth',3,'Color',col(3,:));hold on
xlabel('$t$','Fontsize',20,'Interpreter','latex')
ylabel('d($x_k, X(kh)$)','Fontsize',20,'Interpreter','latex')
l=legend('$h=0.05$','$h=0.4$','$h=0.8$');
l.FontSize = 20;
set(l, 'Interpreter', 'latex','Location','best')

%% Plotting the suboptimality
subplot(1,2,2)
hh1=semilogy(t1,f_val1,'-','Linewidth',3,'Color',col(4,:));hold on
hh4=loglog(t1,(2*zeta*Dist(x1(:,1),x_star)^2)./(h1^2*(1:K1).^2),'--','Linewidth',2,'Color','k');hold on
semilogy(t2,f_val2,'-','Linewidth',3,'Color',col(1,:));hold on
semilogy(t3,f_val3,'-','Linewidth',3,'Color',col(2,:));hold on
semilogy(t4,f_val4,'-','Linewidth',3,'Color',col(3,:));hold on
xlabel('$t$','Fontsize',20,'Interpreter','latex')
ylabel('$f(x)-f(x^*)$','Fontsize',20,'Interpreter','latex')
ll=legend([hh1,hh4],{'ODE solution','$\mathcal{O}(1/t^2)$ bound'});
ll.FontSize = 20;
set(ll, 'Interpreter', 'latex','Location','best')

%% Additional functions for geometry

% Euclidean geometry
function y = RetrEuclidean(x)
	y = x;
end

function y = ExpEuclidean(x,u)
    y=x+u;
end

function d = DistEuclidean(x,y)
    d= norm(x-y);
end

function p = ProjEuclidean(x,u)
    p=u;
end

function u = LogEuclidean(x,y)
    u=y-x;
end

function r = GammaEuclidean(x,y,u)
    r=u;
end

% Hy geometry

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

% Spherical geometry

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






