%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inertial Microcavitation Rheometry (IMR) numerical simulation code
% ====================================================================
% To simulate laser and ultrasound induced inertial cavitation in a
% viscoelastic material. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Welcome to the world of bubble !!!
%
% ====================================================================
% Authors: Carlos Barajas, Eric Johnsen, Jon Estrada, Jin Yang, David Henann
% Contact: jyang526@wisc.edu
% Date: 10/19/2021
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% Parameters to be specified
 
%%%%% Undeformed bubble radius %%%%%
R0 = 50e-6; % Undeformed (equilibrium) bubble radius (m) %%%%% TODO

%%%%% Far-field thermodynamic conditions %%%%%
P_inf = 101325; % Atmospheric pressure (Pa)
T_inf = 298.15; % Far field temperature (K)

%%%%% Type of cavitation driving force %%%%%
%     LIC = Laser induced cavitation
%     UIC = Ultrasound induced cavitation 
cav_type = 'UIC'; %%%%% TODO

%%%%% Driving conditions %%%%%
if strcmp(cav_type,'LIC') == 1
    Rmax = 10*R0; % Maximum (initial) bubble radius for LIC (m) %%%%% TODO
    
    PA = 0; omega = 0; delta = 0; n = 0; % These values won't be used in LIC
    
elseif strcmp(cav_type,'UIC') == 1
    PA = -2.4e6; % Amplitude of the ultrasound pulse (Pa) %%%%% TODO
    omega = 2*pi*(1e6); % Frequency of the ultrasound pulse (1/s) %%%%% TODO
    delta = pi/omega; % Time shift for the ultrasound pulse (s) %%%%% TODO
    n = 3.7; % Exponent that shapes the ultrasound pulse %%%%% TODO
    %
    Rmax = R0; % Assume that initial bubble radius is R0 (stress free)
    
else
    disp('Incorrect cavitation type');
    return;
end

%%%%% Total simulation time %%%%%
tspan = 100e-6; % Total time span (s)

%%%%% Material parameters for the surrounding material %%%%%
G = 2.77e3;     % Ground-state shear modulus (Pa) %%%%% TODO
alpha = 0.48;   % Strain-stiffening parameter (1) (alpha=0: neo-Hookean) %%%%% TODO
mu = 0.186;     % Viscosity (Pa-s) %%%%% TODO
c_long = 1430;  % Longitudinal wave speed (m/s)
rho = 1060;     % Density (kg/m^3)
gamma = 5.6e-2; % Surface tension (N/m)

%%%%%%%%% Please DO NOT modify these parameters unless you know more %%%%%%
%%%%%%%%% accurate information about the bubble contents %%%%%%%%%%%%%%%%%%
%%%%% Parameters for the bubble contents %%%%%
D0 = 24.2e-6;           % Binary diffusion coeff (m^2/s)
kappa = 1.4;            % Specific heats ratio
Ru = 8.3144598;         % Universal gas constant (J/mol-K)
Rv = Ru/(18.01528e-3);  % Gas constant for vapor (Ru/molecular weight) (J/kg-K)
Ra = Ru/(28.966e-3);    % Gas constant for air (Ru/molecular weight) (J/kg-K)
A = 5.28e-5;            % Thermal conductivity parameter (W/m-K^2)
B = 1.17e-2;            % Thermal conductivity parameter (W/m-K)
P_ref = 1.17e11;        % Reference pressure (Pa)
T_ref = 5200;           % Reference temperature (K)

%%%%%%%%% Please DO NOT modify these parameters unless you want to achieve 
%%%%%%%%% higher accuracy in the ode solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Numerical parameters %%%%%
NT = 500; % Mesh points inside the bubble, resolution should be >=500
IMRsolver_RelTolX = 1e-7; % Relative tolerance for the ode solver


%%
% Intermediate calculated variables
%
% General characteristic scales
if strcmp(cav_type,'LIC') == 1
    Rc = Rmax; % Characteristic length scale (m)
    Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
    tc = Rmax/Uc; % Characteristic time scale (s)
elseif strcmp(cav_type,'UIC') == 1
    Rc = R0; % Characteristic length scale (m)
    Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
    tc = R0/Uc; % Characteristic time scale (s)
end

% Parameters for the bubble contents
Pv = P_ref*exp(-T_ref./T_inf); % Vapor pressure evaluated at the far field temperature (Pa)
K_inf = A*T_inf+B; % Thermal conductivity evaluated at the far field temperature (W/m-K)

%%
% Non-dimensional variables
%
C_star = c_long/Uc; % Dimensionless wave speed
We = P_inf*Rc/(2*gamma); % Weber number
Ca = P_inf/G; % Cauchy number
Re = P_inf*Rc/(mu*Uc); % Reynolds number
fom = D0/(Uc*Rc); % Mass Fourier number
chi = T_inf*K_inf/(P_inf*Rc*Uc); % Lockhart–Martinelli number
A_star = A*T_inf/K_inf; % Dimensionless A parameter
B_star = B/K_inf; % Dimensionless B parameter (Note that A_star+B_star=1.)
Pv_star = Pv/P_inf; % Dimensionless vapor saturation pressure at the far field temperature
%
tspan_star = tspan/tc; % Dimensionless time span
%
%%%%% Non-dimensional variable only used for LIC %%%%%
Req = R0/Rmax; % Dimensionless equilibrium bubble radius

%%%%% Non-dimensional variables only used for UIC %%%%%
PA_star = PA/P_inf; % Dimensionless amplitude of the ultrasound pulse
omega_star = omega*tc; % Dimensionless frequency of the ultrasound pulse
delta_star = delta/tc; % Dimensionless time shift for the ultrasound pulse

% Place the necessary quantities in a parameters vector
params = [NT C_star We Ca alpha Re Rv Ra kappa fom chi A_star B_star Pv_star Req PA_star omega_star delta_star n];

%%
% Initial conditions
%
R0_star = 1; % Dimensionless initial bubble radius
U0_star = 0; % Dimensionless initial bubble wall velocity
Theta0 = zeros(1,NT); % Initial dimensionless temperature field
%
if strcmp(cav_type,'LIC') == 1
    P0 = Pv + (P_inf + 2*gamma/R0 - Pv)*((R0/Rmax)^3); % Initial bubble pressure for LIC
    P0_star = P0/P_inf; % Dimensionless initial bubble pressure for LIC
    S0 = (3*alpha-1)*(5 - 4*Req - Req^4)/(2*Ca) + ...
        2*alpha*(27/40 + 1/8*Req^8 + 1/5*Req^5 + 1*Req^2 - 2/Req)/(Ca); % Initial dimensionless elastic stress integral for LIC
    k0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for LIC
elseif strcmp(cav_type,'UIC') == 1
    P0 = P_inf + 2*gamma/R0; % Initial bubble pressure for UIC
    P0_star = P0/P_inf; % Dimensionless initial bubble pressure for UIC
    S0 = 0; % Initial dimensionless elastic stress integral for UIC
    k0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for UIC
end

% Place the initial conditions in the state vector
X0 = [R0_star U0_star P0_star S0 Theta0 k0];
X0 = reshape(X0,length(X0),1);

%% Solve the system of ODEs
%
if strcmp(cav_type,'LIC') == 1
    IMRsolver_InitialStep = [];
elseif strcmp(cav_type,'UIC') == 1
    % Make sure that the initial time step is sufficiently small for UIC
    IMRsolver_InitialStep = delta_star/20; 
end
options = odeset('RelTol',IMRsolver_RelTolX,'InitialStep',IMRsolver_InitialStep);
[t_nondim,X_nondim] = ode23tb(@(t,X) bubble(t,X,cav_type,params),[0 tspan_star], X0, options);

% Extract the solution
R_nondim = X_nondim(:,1); % Bubble wall radius history
U_nondim = X_nondim(:,2); % Bubble wall velocity history
P_nondim = X_nondim(:,3); % Internal bubble pressure history
S_nondim = X_nondim(:,4); % Stress integral history
Theta_nondim = X_nondim(:,5:(NT+4)); % Variable relating to internal temp (theta)
k_nondim = X_nondim(:,(NT+5):(2*NT+4)); % Vapor concentration in the bubble
T_nondim = (A_star - 1 + sqrt(1+2.*A_star.*Theta_nondim))./A_star; % Dimensionless temp in bubble

%%
%%%%% Plot %%%%%
%
figure(1);  
plot(t_nondim,R_nondim,'k-','linewidth',2);
xlabel('$t/t_0$','Interpreter','latex','Fontsize',16);
if strcmp(cav_type,'LIC') == 1
    ylabel('$R/R_{\rm max}$','Interpreter','latex','Fontsize',16);
elseif strcmp(cav_type,'UIC') == 1
    ylabel('$R/R_0$','Interpreter','latex','Fontsize',16);
end
set(gca,'Fontsize',12);

%%%%%% Back to physical world %%%%%%%
t = t_nondim*tc; % unit: s
R = R_nondim*Rc; % unit: m
U = U_nondim*(Rc/tc); % unit: m/s
P = P_nondim*P_inf; % unit: Pa

figure(2)
subplot(2,2,1); plot(t, R , 'k-','linewidth',1); % bubble R vs. t curve
xlabel('$t$ (s)','Interpreter','latex','Fontsize',16); 
ylabel('$R$ (m)','Interpreter','latex','Fontsize',16); 

subplot(2,2,2); plot(t, U , 'k-','linewidth',1); % bubble R,t vs. t curve
xlabel('$t$ (s)','Interpreter','latex','Fontsize',16); 
ylabel('$\dot{R}$ (m/s)','Interpreter','latex','Fontsize',16); 

subplot(2,2,3); plot(t, P , 'k-','linewidth',1); % bubble internal pressure
set(gca,'yscale','log');
xlabel('$t$ (s)','Interpreter','latex','Fontsize',16); 
ylabel('$p$ (Pa)','Interpreter','latex','Fontsize',16); 

subplot(2,2,4); plot(R, U,  'k-','linewidth',1); % bubble dynamics phase diagram
xlabel('$R$ (m)','Interpreter','latex','Fontsize',16); 
ylabel('$\dot{R}$ (m/s)','Interpreter','latex','Fontsize',16); 





%%
%*************************************************************************
% Function that the ODE Solver calls to march governing equations in time
% Unlike in vanilla-IMR this is NOT a nested function

function dxdt = bubble(t,x,cav_type,params)

% Extract quantities from the parameters vector
NT = params(1); % Mesh points inside the bubble
C_star = params(2); % Dimensionless wave speed
We = params(3); % Weber number
Ca = params(4); % Cauchy number
alpha = params(5); % Strain-stiffening parameter
Re = params(6); % Reynolds number
Rv = params(7); % Gas constant for vapor (J/kg-K)
Ra = params(8); % Gas constant for air (J/kg-K)
kappa = params(9); % Specific heats ratio
fom = params(10); % Mass Fourier number
chi = params(11); % Lockhart–Martinelli number
A_star = params(12); % Dimensionless A parameter
B_star = params(13); % Dimensionless B parameter (Note that A_star+B_star=1.)
Pv_star = params(14); % Dimensionless vapor saturation pressure at the far field temperature
Req = params(15); % Dimensionless equilibrium bubble radius (LIC only)
PA_star = params(16); % Dimensionless amplitude of the ultrasound pulse (UIC only)
omega_star = params(17); % Dimensionless frequency of the ultrasound pulse (UIC only)
delta_star = params(18); % Dimensionless time shift for the ultrasound pulse (UIC only)
n = params(19); % Exponent that shapes the ultrasound pulse (UIC only)

% Extract quantities from the state vector
R = x(1); % Bubble wall radius
U = x(2); % Bubble wall velocity
P = x(3); % Internal bubble pressure
S = x(4); % Stress integral
Theta = x(5:(NT+4)); % Variable relating to internal temp (theta)
k = x((NT+5):(2*NT+4)); % Vapor mass fraction (k)

%******************************************
% Set up grid inside the bubble
deltaY = 1/(NT-1); % Dimensionless grid spacing inside the bubble
ii = 1:1:NT;
yk = ((ii-1)*deltaY)'; % Dimensionless grid points inside the bubble
%******************************************

%******************************************
% Apply the Dirichlet BC for the vapor mass fraction at the bubble wall
k(end) = (1+(Rv/Ra)*(P/Pv_star-1))^(-1);
%******************************************

%******************************************
% Calculate mixture fields inside the bubble
T = (A_star - 1 + sqrt(1+2.*A_star.*Theta))./A_star; % Dimensionless temperature T/T_inf
K_star = A_star.*T+B_star; % Dimensionless mixture thermal conductivity field
Rmix = k.*Rv + (1-k).*Ra; % Mixture gas constant field (J/kg-K)
%******************************************

%******************************************
% Calculate spatial derivatives of the temp and vapor conc fields
DTheta = [0; % Neumann BC at origin
    (Theta(3:end)-Theta(1:end-2))/(2*deltaY); % Central difference approximation for interior points
    (3*Theta(end)-4*Theta(end-1)+Theta(end-2))/(2*deltaY)]; % Backward difference approximation at the bubble wall
DDTheta = [6*(Theta(2)-Theta(1))/deltaY^2; % Laplacian in spherical coords at the origin obtained using L'Hopital's rule
    (diff(diff(Theta)/deltaY)/deltaY + (2./yk(2:end-1)).*DTheta(2:end-1)); % Central difference approximation for Laplacian in spherical coords
    ((2*Theta(end)-5*Theta(end-1)+4*Theta(end-2)-Theta(end-3))/deltaY^2+(2/yk(end))*DTheta(end))]; % Laplacian at the bubble wall does not affect the solution
Dk = [0; % Neumann BC at origin
    (k(3:end)-k(1:end-2))/(2*deltaY); % Central difference approximation for interior points
    (3*k(end)-4*k(end-1)+k(end-2))/(2*deltaY)]; % Backward difference approximation at the bubble wall
DDk = [6*(k(2)-k(1))/deltaY^2; % Laplacian in spherical coords at the origin obtained using L'Hopital's rule
    (diff(diff(k)/deltaY)/deltaY + (2./yk(2:end-1)).*Dk(2:end-1)); % Central difference approximation for Laplacian in spherical coords
    ((2*k(end)-5*k(end-1)+4*k(end-2)-k(end-3))/deltaY^2+(2/yk(end))*Dk(end))]; % Laplacian at the bubble wall does not affect the solution
%******************************************

%******************************************
% Internal bubble pressure evolution equation
pdot = 3/R*(-kappa*P*U + (kappa-1)*chi*DTheta(end)/R ...
    + kappa*P*fom*Rv*Dk(end)/(R*Rmix(end)*(1-k(end))));
%******************************************

%******************************************
% Dimensionless mixture velocity field inside the bubble
Umix = ((kappa-1).*chi./R.*DTheta-R.*yk.*pdot./3)./(kappa.*P) + fom./R.*(Rv-Ra)./Rmix.*Dk;
%******************************************

%******************************************
% Evolution equation for the temperature (theta) of the mixture inside the bubble
Theta_prime = (pdot + (DDTheta).*chi./R.^2).*(K_star.*T./P.*(kappa-1)./kappa) ...
    - DTheta.*(Umix-yk.*U)./R ...
    + fom./(R.^2).*(Rv-Ra)./Rmix.*Dk.*DTheta;
Theta_prime(end) = 0; % Dirichlet BC at the bubble wall
%******************************************

%******************************************
% Evolution equation for the vapor concentration inside the bubble
k_prime = fom./R.^2.*(DDk + Dk.*(-((Rv - Ra)./Rmix).*Dk - DTheta./sqrt(1+2.*A_star.*Theta)./T)) ...
    - (Umix-U.*yk)./R.*Dk;
k_prime(end) = 0; % Dirichlet BC at the bubble wall
%******************************************

%******************************************
% Elastic stress in the material
%     (viscous contribution is accounted for in the Keller-Miksis equation)
if strcmp(cav_type,'LIC') == 1
    Rst = R/Req;
    Sdot = 2*U/R*(3*alpha-1)*(1/Rst + 1/Rst^4)/Ca - ...
        2*alpha*U/R*(1/Rst^8 + 1/Rst^5 + 2/Rst^2 + 2*Rst)/(Ca);
elseif strcmp(cav_type,'UIC') == 1
    Sdot = 2*U/R*(3*alpha-1)*(1/R + 1/R^4)/Ca - ...
        2*alpha*U/R*(1/R^8 + 1/R^5 + 2/R^2 + 2*R)/(Ca);
end
%******************************************

%******************************************
% External pressure for UIC
if strcmp(cav_type,'LIC') == 1
    Pext = 0; % No external pressure for LIC
    Pextdot = 0;
elseif strcmp(cav_type,'UIC') == 1
    if (abs(t-delta_star)>(pi/omega_star))
        Pext = 0;
        Pextdot = 0;
    else
        Pext = PA_star*((1+cos(omega_star*(t-delta_star)))/2)^n;
        Pextdot = (-omega_star*n*PA_star/2)*(((1+cos(omega_star*(t-delta_star)))/2)^(n-1))*sin(omega_star*(t-delta_star));
    end
end
%******************************************

%******************************************
% Keller-Miksis equations
rdot = U;
udot = ((1+U/C_star)*(P - 1/(We*R) + S - 4*U/(Re*R) - 1 - Pext)  ...
    + R/C_star*(pdot + U/(We*R^2) + Sdot + 4*U^2/(Re*R^2) - Pextdot) ...
    - (3/2)*(1-U/(3*C_star))*U^2)/((1-U/C_star)*R + 4/(C_star*Re));
%******************************************

dxdt = [rdot; udot; pdot; Sdot; Theta_prime; k_prime];

end