using DifferentialEquations
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using Measures

global_logger(TerminalLogger())

# Define a structure to hold all parameters
struct Params
    cav_type::String
    NT::Int
    C_star::Float64
    We::Float64
    Ca::Float64
    alpha::Float64
    Re::Float64
    Rv::Float64
    Ra::Float64
    kappa::Float64
    fom::Float64
    chi::Float64
    A_star::Float64
    B_star::Float64
    Pv_star::Float64
    Req::Float64
    PA_star::Float64
    omega_star::Float64
    delta_star::Float64
    n::Float64
end

function main()

    # **Parameters to be specified**

    # Undeformed bubble radius
    R0 = 50e-6 # Undeformed (equilibrium) bubble radius (m)

    # Far-field thermodynamic conditions
    P_inf = 101325.0 # Atmospheric pressure (Pa)
    T_inf = 298.15    # Far field temperature (K)

    # Type of cavitation driving force
    cav_type = "LIC" # "LIC" or "UIC"

    # **Driving conditions**
    if cav_type == "LIC"
        Rmax = 10.0 * R0 # Maximum (initial) bubble radius for LIC (m)
        PA = 0.0
        omega = 0.0
        delta = 0.0
        n = 0.0
    elseif cav_type == "UIC"
        PA = -2.4e6              # Amplitude of the ultrasound pulse (Pa)
        omega = 2 * π * 1.0e6    # Frequency of the ultrasound pulse (1/s)
        delta = π / omega        # Time shift for the ultrasound pulse (s)
        n = 3.7                  # Exponent that shapes the ultrasound pulse
        Rmax = R0
    else
        throw(ArgumentError("Incorrect cavitation type"))
    end

    # Total simulation time
    tspan = 100e-6 # Total time span (s)

    # **Material parameters**
    G = 2.77e3             # Ground-state shear modulus (Pa)
    alpha = 0.48           # Strain-stiffening parameter
    mu = 0.186             # Viscosity (Pa-s)
    c_long = 1430.0        # Longitudinal wave speed (m/s)
    rho = 1060.0           # Density (kg/m^3)
    gamma = 5.6e-2         # Surface tension (N/m)

    # **Bubble contents parameters**
    D0 = 24.2e-6           # Binary diffusion coeff (m²/s)
    kappa = 1.4            # Specific heats ratio
    Ru = 8.3144598         # Universal gas constant (J/mol-K)
    Rv = Ru / 0.01801528   # Gas constant for vapor
    Ra = Ru / 0.028966     # Gas constant for air
    A = 5.28e-5            # Thermal conductivity parameter
    B = 1.17e-2            # Thermal conductivity parameter
    P_ref = 1.17e11        # Reference pressure (Pa)
    T_ref = 5200.0         # Reference temperature (K)

    # **Numerical parameters**
    NT = 500               # Mesh points inside bubble
    IMRsolver_RelTolX = 1e-10 # Relative tolerance for ODE solver

    # **Calculate characteristic scales**
    if cav_type == "LIC"
        Rc = Rmax
    else
        Rc = R0
    end

    Uc = sqrt(P_inf / rho)
    tc = Rc / Uc

    # Parameters for bubble contents
    Pv = P_ref * exp(-T_ref / T_inf)
    K_inf = A * T_inf + B

    # **Non-dimensional variables**
    C_star = c_long / Uc
    We = P_inf * Rc / (2.0 * gamma)
    Ca = P_inf / G
    Re = P_inf * Rc / (mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = T_inf * K_inf / (P_inf * Rc * Uc)
    A_star = (A * T_inf) / K_inf
    B_star = B / K_inf
    Pv_star = Pv / P_inf
    tspan_star = tspan / tc

    # **Additional non-dimensional variables**
    Req = (cav_type == "LIC") ? (R0 / Rmax) : 1.0
    PA_star = PA / P_inf
    omega_star = omega * tc
    delta_star = delta / tc

    # Initialize parameters structure
    params = Params(
        cav_type,
        NT,
        C_star,
        We,
        Ca,
        alpha,
        Re,
        Rv,
        Ra,
        kappa,
        fom,
        chi,
        A_star,
        B_star,
        Pv_star,
        Req,
        PA_star,
        omega_star,
        delta_star,
        n
    )

    # **Initial conditions setup**
    R0_star = 1.0
    U0_star = 0.0
    Theta0 = zeros(NT)

    if cav_type == "LIC"
        P0 = Pv + (P_inf + 2.0 * gamma / R0 - Pv) * ((R0 / Rmax)^3)
        P0_star = P0 / P_inf
        S0 = ((3.0 * alpha - 1.0) * (5.0 - 4.0 * params.Req - params.Req^4)) / (2.0 * Ca) +
             (2.0 * alpha * (27.0/40.0 + 1.0/8.0 * params.Req^8 + 1.0/5.0 * params.Req^5 +
              params.Req^2 - 2.0 / params.Req)) / Ca
    else
        P0 = P_inf + 2.0 * gamma / R0
        P0_star = P0 / P_inf
        S0 = 0.0
    end

    k0 = ones(NT) .* ((1.0 + (params.Rv / params.Ra) .* (P0_star ./ params.Pv_star .- 1.0)).^(-1.0))
    X0 = vcat(R0_star, U0_star, P0_star, S0, Theta0, k0)

    # **Define ODE problem**
    prob = ODEProblem(bubble, X0, (0.0, tspan_star), params)

function print_step(integrator)
    println("Time: $(integrator.t), Radius: $(integrator.u[1])")
    u_modified!(integrator, false)
end

    step_cb = DiscreteCallback((u,t,integrator)->true, print_step)

    # **Solve ODE system**
    sol = solve(prob, QBDF(autodiff=false), reltol=IMRsolver_RelTolX, callback=step_cb,saveat=range(0.0, stop=tspan_star, length=1000), progress=true)

    # **Extract solutions**
    t_nondim = sol.t
    X_nondim = reduce(hcat, sol.u)'
    R_nondim = X_nondim[:, 1]
    U_nondim = X_nondim[:, 2]
    P_nondim = X_nondim[:, 3]

    # **Convert to physical units**
    t_phys = t_nondim .* tc
    R_phys = R_nondim .* Rc
    U_phys = U_nondim .* (Rc / tc)
    P_phys = P_nondim .* P_inf

    # **Plotting**
    p1 = plot(t_phys, R_phys, label="R (m)", linewidth=1,
        xlabel="t (s)", ylabel="R (m)")
    p2 = plot(t_phys, U_phys, label="Ṙ (m/s)", linewidth=1,
        xlabel="t (s)", ylabel="Ṙ (m/s)")
    p3 = plot(t_phys, P_phys, label="p (Pa)", linewidth=1, yaxis=:log10,
        xlabel="t (s)", ylabel="p (Pa)")
    p4 = plot(R_phys, U_phys, label="Ṙ vs R", linewidth=1,
        xlabel="R (m)", ylabel="Ṙ (m/s)")

    # Create combined plot
    combined_plot = plot(
        p1, p2, p3, p4,
        layout = (2, 2),
        size = (1200, 800),
        margin = 5
    )

    # Save individual plots
    savefig(p1, "radius_vs_time.png")
    savefig(p2, "velocity_vs_time.png")
    savefig(p3, "pressure_vs_time.png")
    savefig(p4, "phase_portrait.png")

    # Save combined plot
    savefig(combined_plot, "combined_plots.png")

    # Display the plot
    display(combined_plot)

end

# **Define the ODE function with correct signature (u, p, t)**
function bubble(u, p::Params, t::Float64)
    # **Extract parameters**
    NT = p.NT
    C_star = p.C_star
    We = p.We
    Ca = p.Ca
    alpha = p.alpha
    Re = p.Re
    Rv = p.Rv
    Ra = p.Ra
    kappa = p.kappa
    fom = p.fom
    chi = p.chi
    A_star = p.A_star
    B_star = p.B_star
    Pv_star = p.Pv_star
    Req = p.Req
    PA_star = p.PA_star
    omega_star = p.omega_star
    delta_star = p.delta_star
    n = p.n
    cav_type = p.cav_type

    # **Extract state variables**
    R = u[1]                             # Bubble wall radius
    U = u[2]                             # Bubble wall velocity
    P_int = u[3]                         # Internal bubble pressure
    S = u[4]                             # Stress integral
    Theta = u[5:NT+4]                    # Temperature variable
    k = u[NT+5:end]                      # Vapor mass fraction

    # **Set up grid inside bubble**
    deltaY = 1.0 / (NT - 1)
    yk = LinRange(0.0, 1.0, NT)

    # **Apply Dirichlet BC for vapor mass fraction (outer boundary)**
    # k[end] is enforced by equilibrium condition
    k[end] = (1.0 + (p.Rv / p.Ra) * (P_int / Pv_star - 1.0))^(-1.0)

    # **Calculate mixture fields**
    T = (A_star .- 1.0 .+ sqrt.(1.0 .+ 2.0 .* A_star .* Theta)) ./ A_star
    K_star = A_star .* T .+ B_star
    Rmix = k .* p.Rv .+ (1.0 .- k) .* p.Ra

    # **Compute first derivatives (DTheta and Dk)**
    DTheta = fill!(Theta,0.0)
    Dk = fill!(k,0.0)

    # Interior points (central difference)
    @inbounds for i in 2:NT-1
        DTheta[i] = (Theta[i+1] - Theta[i-1]) / (2.0 * deltaY)
        Dk[i] = (k[i+1] - k[i-1]) / (2.0 * deltaY)
    end

    # Boundary points
    DTheta[1] = 0.0 # Neumann BC at origin
    DTheta[end] = (3.0 * Theta[end] - 4.0 * Theta[end-1] + Theta[end-2]) / (2.0 * deltaY)
    Dk[1] = 0.0
    Dk[end] = (3.0 * k[end] - 4.0 * k[end-1] + k[end-2]) / (2.0 * deltaY)

    # **Compute second derivatives (Laplacian in spherical symmetry)**
    DDTheta = fill!(Theta,0.0)
    DDk = fill!(k,0.0)

    # At the center (i=1)
    DDTheta[1] = 6.0 * (Theta[2] - Theta[1]) / (deltaY^2)
    DDk[1] = 6.0 * (k[2] - k[1]) / (deltaY^2)

    # Interior points
    @inbounds for i in 2:NT-1
        DDTheta[i] = (Theta[i+1] - 2.0 * Theta[i] + Theta[i-1]) / (deltaY^2) +
                     (2.0 / yk[i]) * DTheta[i]
        DDk[i] = (k[i+1] - 2.0 * k[i] + k[i-1]) / (deltaY^2) +
                 (2.0 / yk[i]) * Dk[i]
    end

    # Outer boundary (i=NT)
    DDTheta[end] = (2.0 * Theta[end] - 5.0 * Theta[end-1] + 4.0 * Theta[end-2] - Theta[end-3]) / (deltaY^2) +
                   (2.0 / yk[end]) * DTheta[end]
    DDk[end] = (2.0 * k[end] - 5.0 * k[end-1] + 4.0 * k[end-2] - k[end-3]) / (deltaY^2) +
               (2.0 / yk[end]) * Dk[end]

    # **Pressure evolution (pdot)**
    pdot = (3.0 / R) * (
           -kappa * P_int * U +
            (kappa - 1.0) * chi * DTheta[end] / R +
            kappa * P_int * fom * p.Rv * Dk[end] / (R * Rmix[end] * (1.0 - k[end]))
    )

    # **Mixture velocity field (Umix)**
    Umix = ((kappa - 1.0) .* chi ./ R .* DTheta .- R .* yk .* pdot ./ 3.0) ./ (kappa .* P_int) .+
            fom ./ R .* (p.Rv .- p.Ra) ./ Rmix .* Dk

    # **Temperature evolution (Theta_prime)**
    Theta_prime = (pdot .+ DDTheta .* chi ./ R.^2) .* (K_star .* T ./ P_int .* (kappa - 1.0) ./ kappa) .-
                  DTheta .* (Umix .- yk .* U) ./ R .+
                  fom ./ R.^2 .* (p.Rv .- p.Ra) ./ Rmix .* Dk .* DTheta

    # Enforce boundary condition: no temperature flux at outer boundary
    Theta_prime[end] = 0.0

    # **Vapor concentration evolution (k_prime)**
    k_prime = fom ./ R.^2 .* (DDk .+ Dk .* (-((p.Rv .- p.Ra) ./ Rmix) .* Dk .- DTheta ./ sqrt.(1.0 .+ 2.0 .* A_star .* Theta) ./ T )) .- 
              (Umix .- U .* yk) ./ R .* Dk

    # Enforce boundary condition
    k_prime[end] = 0.0

    # **Elastic stress (Sdot)**
    if cav_type == "LIC"
        Rst = R / Req
        Sdot = (2.0 * U / R) * (3.0 * alpha - 1.0) * (1.0 / Rst + 1.0 / Rst^4) / Ca -
               (2.0 * alpha * U / R) * (1.0 / Rst^8 + 1.0 / Rst^5 + 2.0 / Rst^2 + 2.0 * Rst) / Ca
    else
        Sdot = (2.0 * U / R) * (3.0 * alpha - 1.0) * (1.0 / R .+ 1.0 ./ R.^4) / Ca -
               (2.0 * alpha * U / R) * (1.0 ./ R.^8 .+ 1.0 ./ R.^5 .+ 2.0 ./ R.^2 .+ 2.0 .* R) / Ca
    end

    # **External driving for UIC (Pressure)**
    if cav_type == "LIC"
        Pext = 0.0
        Pextdot = 0.0
    else
        if abs(t - delta_star) > (π / omega_star)
            Pext = 0.0
            Pextdot = 0.0
        else
            Pext = PA_star * ((1.0 .+ cos.(omega_star * (t - delta_star))) ./ 2.0).^n
            Pextdot = (-omega_star * n * PA_star / 2.0) .* ((1.0 .+ cos.(omega_star * (t - delta_star))) ./ 2.0).^(n .- 1.0) .* sin.(omega_star * (t - delta_star))
        end
    end

    # **Keller-Miksis equation to get rdot and udot**
    rdot = U

    numerator = (1.0 .+ U ./ C_star) .* (P_int .- 1.0 ./ (We .* R) .+ S .- 4.0 .* U ./ (Re .* R) .- 1.0 .- Pext) .+
                (R ./ C_star) .* (pdot .+ U ./ (We .* R.^2) .+ Sdot .+ 4.0 .* U.^2 ./ (Re .* R.^2) .- Pextdot) .-
                (3.0 ./ 2.0) .* (1.0 .- U ./ (3.0 .* C_star)) .* U.^2

    denominator = (1.0 .- U ./ C_star) .* R .+ 4.0 ./ (C_star .* Re)
    udot = numerator ./ denominator

    # **Assemble derivatives into a single vector**
    du = zeros(length(u))
    du[1] = rdot
    du[2] = udot
    du[3] = pdot
    du[4] = Sdot
    du[5:NT+4] = Theta_prime
    du[NT+5:end] = k_prime

    return du
end
