import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def main():
    # Parameters to be specified
    # Undeformed bubble radius
    R0 = 50e-6  # Undeformed (equilibrium) bubble radius (m)

    # Far-field thermodynamic conditions
    P_inf = 101325  # Atmospheric pressure (Pa)
    T_inf = 298.15  # Far field temperature (K)

    # Type of cavitation driving force
    cav_type = 'LIC'  # 'LIC' or 'UIC'

    # Driving conditions
    if cav_type == 'LIC':
        Rmax = 10 * R0  # Maximum (initial) bubble radius for LIC (m)
        PA = 0
        omega = 0
        delta = 0
        n = 0
    elif cav_type == 'UIC':
        PA = -2.4e6  # Amplitude of the ultrasound pulse (Pa)
        omega = 2 * np.pi * (1e6)  # Frequency of the ultrasound pulse (1/s)
        delta = np.pi / omega  # Time shift for the ultrasound pulse (s)
        n = 3.7  # Exponent that shapes the ultrasound pulse
        Rmax = R0
    else:
        raise ValueError('Incorrect cavitation type')

    # Total simulation time
    tspan = 100e-6  # Total time span (s)

    # Material parameters
    G = 2.77e3  # Ground-state shear modulus (Pa)
    alpha = 0.48  # Strain-stiffening parameter
    mu = 0.186  # Viscosity (Pa-s)
    c_long = 1430  # Longitudinal wave speed (m/s)
    rho = 1060  # Density (kg/m^3)
    gamma = 5.6e-2  # Surface tension (N/m)

    # Bubble contents parameters
    D0 = 24.2e-6  # Binary diffusion coeff (m^2/s)
    kappa = 1.4  # Specific heats ratio
    Ru = 8.3144598  # Universal gas constant (J/mol-K)
    Rv = Ru / (18.01528e-3)  # Gas constant for vapor
    Ra = Ru / (28.966e-3)  # Gas constant for air
    A = 5.28e-5  # Thermal conductivity parameter
    B = 1.17e-2  # Thermal conductivity parameter
    P_ref = 1.17e11  # Reference pressure (Pa)
    T_ref = 5200  # Reference temperature (K)

    # Numerical parameters
    NT = 500  # Mesh points inside bubble
    IMRsolver_RelTolX = 1e-10  # Relative tolerance for ODE solver

    # Calculate characteristic scales
    if cav_type == 'LIC':
        Rc = Rmax
        Uc = np.sqrt(P_inf / rho)
        tc = Rmax / Uc
    else:
        Rc = R0
        Uc = np.sqrt(P_inf / rho)
        tc = R0 / Uc

    # Parameters for bubble contents
    Pv = P_ref * np.exp(-T_ref / T_inf)
    K_inf = A * T_inf + B

    # Non-dimensional variables
    C_star = c_long / Uc
    We = P_inf * Rc / (2 * gamma)
    Ca = P_inf / G
    Re = P_inf * Rc / (mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = T_inf * K_inf / (P_inf * Rc * Uc)
    A_star = A * T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / P_inf
    tspan_star = tspan / tc

    # Additional non-dimensional variables
    Req = R0 / Rmax if cav_type == 'LIC' else 1
    PA_star = PA / P_inf
    omega_star = omega * tc
    delta_star = delta / tc

    params = [NT, C_star, We, Ca, alpha, Re, Rv, Ra, kappa, fom, chi, 
              A_star, B_star, Pv_star, Req, PA_star, omega_star, delta_star, n]

    # Initial conditions setup
    R0_star = 1
    U0_star = 0
    Theta0 = np.zeros(NT)

    if cav_type == 'LIC':
        P0 = Pv + (P_inf + 2 * gamma / R0 - Pv) * ((R0 / Rmax) ** 3)
        P0_star = P0 / P_inf
        S0 = (3 * alpha - 1) * (5 - 4 * Req - Req ** 4) / (2 * Ca) + \
             2 * alpha * (27/40 + 1/8 * Req ** 8 + 1/5 * Req ** 5 + Req ** 2 - 2/Req) / Ca
    else:
        P0 = P_inf + 2 * gamma / R0
        P0_star = P0 / P_inf
        S0 = 0

    k0 = np.ones(NT) * ((1 + (Rv/Ra) * (P0_star/Pv_star - 1)) ** (-1))
    X0 = np.concatenate(([R0_star, U0_star, P0_star, S0], Theta0, k0))

    # Solve ODE system
    t_eval = np.linspace(0, tspan_star, 1000)
    sol = solve_ivp(
        lambda t, X: bubble(t, X, cav_type, params),
        [0, tspan_star],
        X0,
        method='BDF',
        rtol=IMRsolver_RelTolX,
        t_eval=t_eval
    )

    # Extract solutions
    t_nondim = sol.t
    X_nondim = sol.y.T
    R_nondim = X_nondim[:, 0]
    U_nondim = X_nondim[:, 1]
    P_nondim = X_nondim[:, 2]

    # Convert to physical units
    t = t_nondim * tc
    R = R_nondim * Rc
    U = U_nondim * (Rc/tc)
    P = P_nondim * P_inf

    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, R, 'k-', linewidth=1)
    plt.xlabel('t (s)')
    plt.ylabel('R (m)')

    plt.subplot(2, 2, 2)
    plt.plot(t, U, 'k-', linewidth=1)
    plt.xlabel('t (s)')
    plt.ylabel('Ṙ (m/s)')

    plt.subplot(2, 2, 3)
    plt.plot(t, P, 'k-', linewidth=1)
    plt.yscale('log')
    plt.xlabel('t (s)')
    plt.ylabel('p (Pa)')

    plt.subplot(2, 2, 4)
    plt.plot(R, U, 'k-', linewidth=1)
    plt.xlabel('R (m)')
    plt.ylabel('Ṙ (m/s)')

    plt.tight_layout()
    plt.show()

def bubble(t, x, cav_type, params):
    # Extract parameters
    NT, C_star, We, Ca, alpha, Re, Rv, Ra, kappa, fom, chi, A_star, B_star, \
    Pv_star, Req, PA_star, omega_star, delta_star, n = params
    
    # Extract state variables
    R = x[0]  # Bubble wall radius
    U = x[1]  # Bubble wall velocity
    P = x[2]  # Internal bubble pressure
    S = x[3]  # Stress integral
    Theta = x[4:NT+4]  # Temperature variable
    k = x[NT+4:2*NT+4]  # Vapor mass fraction
    
    # Set up grid inside bubble
    deltaY = 1/(NT-1)
    yk = np.linspace(0, 1, NT)
    
    # Apply Dirichlet BC for vapor mass fraction
    k[-1] = (1 + (Rv/Ra)*(P/Pv_star-1))**(-1)
    
    # Calculate mixture fields
    T = (A_star - 1 + np.sqrt(1 + 2*A_star*Theta))/A_star
    K_star = A_star*T + B_star
    Rmix = k*Rv + (1-k)*Ra
    
    # Calculate spatial derivatives
    DTheta = np.zeros_like(Theta)
    DDTheta = np.zeros_like(Theta)
    Dk = np.zeros_like(k)
    DDk = np.zeros_like(k)
    
    # Interior points
    DTheta[1:-1] = (Theta[2:] - Theta[:-2])/(2*deltaY)
    Dk[1:-1] = (k[2:] - k[:-2])/(2*deltaY)
    
    # Boundary points
    DTheta[0] = 0  # Neumann BC at origin
    DTheta[-1] = (3*Theta[-1] - 4*Theta[-2] + Theta[-3])/(2*deltaY)
    Dk[0] = 0
    Dk[-1] = (3*k[-1] - 4*k[-2] + k[-3])/(2*deltaY)
    
    # Laplacian terms
    DDTheta[0] = 6*(Theta[1] - Theta[0])/deltaY**2
    DDTheta[1:-1] = (np.diff(np.diff(Theta))/deltaY**2 + (2/yk[1:-1])*DTheta[1:-1])
    DDTheta[-1] = (2*Theta[-1] - 5*Theta[-2] + 4*Theta[-3] - Theta[-4])/deltaY**2 + (2/yk[-1])*DTheta[-1]
    
    DDk[0] = 6*(k[1] - k[0])/deltaY**2
    DDk[1:-1] = (np.diff(np.diff(k))/deltaY**2 + (2/yk[1:-1])*Dk[1:-1])
    DDk[-1] = (2*k[-1] - 5*k[-2] + 4*k[-3] - k[-4])/deltaY**2 + (2/yk[-1])*Dk[-1]
    
    # Pressure evolution
    pdot = 3/R*(-kappa*P*U + (kappa-1)*chi*DTheta[-1]/R + 
                kappa*P*fom*Rv*Dk[-1]/(R*Rmix[-1]*(1-k[-1])))
    
    # Mixture velocity field
    Umix = ((kappa-1)*chi/R*DTheta - R*yk*pdot/3)/(kappa*P) + fom/R*(Rv-Ra)/Rmix*Dk
    
    # Temperature evolution
    Theta_prime = (pdot + DDTheta*chi/R**2)*(K_star*T/P*(kappa-1)/kappa) - \
                 DTheta*(Umix-yk*U)/R + fom/R**2*(Rv-Ra)/Rmix*Dk*DTheta
    Theta_prime[-1] = 0
    
    # Vapor concentration evolution
    k_prime = fom/R**2*(DDk + Dk*(-((Rv - Ra)/Rmix)*Dk - 
              DTheta/np.sqrt(1+2*A_star*Theta)/T)) - (Umix-U*yk)/R*Dk
    k_prime[-1] = 0
    
    # Elastic stress
    if cav_type == 'LIC':
        Rst = R/Req
        Sdot = 2*U/R*(3*alpha-1)*(1/Rst + 1/Rst**4)/Ca - \
               2*alpha*U/R*(1/Rst**8 + 1/Rst**5 + 2/Rst**2 + 2*Rst)/Ca
    else:
        Sdot = 2*U/R*(3*alpha-1)*(1/R + 1/R**4)/Ca - \
               2*alpha*U/R*(1/R**8 + 1/R**5 + 2/R**2 + 2*R)/Ca
    
    # External pressure for UIC
    if cav_type == 'LIC':
        Pext = 0
        Pextdot = 0
    else:
        if abs(t-delta_star) > (np.pi/omega_star):
            Pext = 0
            Pextdot = 0
        else:
            Pext = PA_star*((1+np.cos(omega_star*(t-delta_star)))/2)**n
            Pextdot = (-omega_star*n*PA_star/2)*(((1+np.cos(omega_star*(t-delta_star)))/2)**(n-1))*\
                      np.sin(omega_star*(t-delta_star))
    
    # Keller-Miksis equations
    rdot = U
    udot = ((1+U/C_star)*(P - 1/(We*R) + S - 4*U/(Re*R) - 1 - Pext) + 
            R/C_star*(pdot + U/(We*R**2) + Sdot + 4*U**2/(Re*R**2) - Pextdot) - 
            (3/2)*(1-U/(3*C_star))*U**2)/((1-U/C_star)*R + 4/(C_star*Re))
    
    return np.concatenate(([rdot, udot, pdot, Sdot], Theta_prime, k_prime))

if __name__ == "__main__":
    main()

