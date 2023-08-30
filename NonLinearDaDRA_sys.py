import dadra

# define the dynamics of the system
def lorenz_system(z, t, disturbance=None, eta=1.0, sigma=10.0, rho=28.0, beta=8 / 3):
    dzdt = [
        sigma * (z[1] - z[0]) + disturbance.get_dist(0, t) * eta,
        z[0] * (rho - z[2]) - z[1] + disturbance.get_dist(1, t) * eta,
        z[0] * z[1] - beta * z[2] + disturbance.get_dist(2, t) * eta,
    ]
    return dzdt

def cstrdiscr(x, u, dt, rho = 1000, Cp = 0.239, deltaH = -5e4, E_R = 8750, k0 = 7.2e10, UA = 5e4, q = 100, Tf = 350, V = 100, C_Af = 1, C_A0 = 0.5, T_0 = 350, T_c0 = 300):
    """
        discrete-time version of the stirred-tank reactor system
    """
    U = np.matmul(np.array([-3, -6.9]), x)
    x_temp = np.array([x[0, 0] + C_A0, x[1, 0]+T_0]).reshape(2, 1)
    U = U + np.array([T_c0])
    f1 = ((1-(q*dt)/(2*V) - k0*dt*np.exp(-E_R /
          x_temp[1, 0]))*x_temp[0, 0] + q/V * C_Af * dt)/(1 + (q*dt)/(2*V)) + u[0]*dt
    f2 = (x_temp[1, 0]*(1-0.5*dt - (dt*UA)/(2*V*rho*Cp)) + dt*(Tf*q/V + (UA*U)/(V*rho*Cp)) - x_temp[0, 0] *
          (deltaH*k0*dt)/(rho*Cp) * np.exp(-E_R/x_temp[1, 0])) / (1+0.5*dt*q/V+(dt*UA)/(2*V*rho*Cp)) + u[1, 0]*dt
    f1 = f1 - C_A0
    f2 = f2 - T_0
    return np.array([f1, f2])

# define the intervals for the initial states of the variables in the system
l_state_dim = 3
l_intervals = [(0, 1) for i in range(l_state_dim)]
dist_list = [dadra.disturbance.ScalarDisturbance.sin_disturbance(1) for _ in range(l_state_dim)]
l_disturbance = dadra.disturbance.Disturbance(dist_list)

# instantiate a DisturbedSystem object for a disturbed system
l_ds = dadra.DisturbedSystem(
    dyn_func=lorenz_system,
    intervals=l_intervals,
    state_dim=l_state_dim,
    disturbance=l_disturbance,
    timesteps=100,
    parts=1001,
)

# instantiate an Estimator object
l_e = dadra.Estimator(dyn_sys=l_ds, epsilon=0.05, delta=1e-9, p=2, normalize=True)

# print out a summary of the Estimator object
l_e.summary()


# make a reachable set estimate on the disturbed system
l_e.estimate()


# print out a summary of the Estimator object once more


# save a plot of the 2D contours of the estimated reachable set
l_e.plot_2D_cont("figures/sin_estimate_2D.png", grid_n=200)


# save a plot and a rotating gif of the 3D contours of the estimated reachable set
l_e.plot_3D_cont(
    "figures/sin_estimate_3D.png", grid_n=100, gif_name="figures/sin_estimate_3D.gif"
)