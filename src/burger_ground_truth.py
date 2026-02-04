import numpy as np

def burgers_ground_truth_fd(nu=0.01, Nx=800, Nt=6000, x_min=-1.0, x_max=1.0, t_max=1.0):
    """
    Ground truth for viscous Burgers using finite differences with periodic BC.
    PDE: u_t + u u_x = nu u_xx
    IC:  u(x,0) = -sin(pi x)
    """
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # periodic grid
    dx = (x_max - x_min) / Nx
    t = np.linspace(0.0, t_max, Nt)
    dt = t_max / (Nt - 1)

    # initial condition
    u = -np.sin(np.pi * x)
    U = np.zeros((Nt, Nx), dtype=np.float64)
    U[0] = u.copy()

    # Stability-ish note: explicit diffusion wants dt <= dx^2/(2*nu).
    # We'll keep dt small via Nt.

    for n in range(1, Nt):
        # periodic shifts
        u_roll_p = np.roll(u, -1)  # u_{j+1}
        u_roll_m = np.roll(u,  1)  # u_{j-1}

        # derivatives (2nd-order central)
        ux  = (u_roll_p - u_roll_m) / (2.0 * dx)
        uxx = (u_roll_p - 2.0*u + u_roll_m) / (dx*dx)

        # time update (explicit)
        u = u + dt * (-u * ux + nu * uxx)

        U[n] = u

    return x, t, U