import numpy as np
import matplotlib.pyplot as plt

try:
    from project.full_heat_eqn import heat_equation_solver_manufactured_solution, u_0_step
    from project.diff_2d import apply_bcs, I, plot2D
    from project.utils import plot_2D_animation
except Exception:
    from full_heat_eqn import heat_equation_solver_manufactured_solution
    from diff_2d import apply_bcs, I, plot2D
    from utils import plot_2D_animation

if __name__ == '__main__':
    a, b = 0, 2 * np.pi
    n = 20

    N = (n + 1) ** 2
    t0 = 0  # sek t start
    T = 20  # sek t stlutt
    theta = 0

    # Exact function
    def u_func(x, y, t, pck=np):
        k, l = 1, 1
        mu = k ** 2 + l ** 2
        return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)

    def g(x, y, t):
        return x * y * 0


    for alpha in [0.99, 1, 1.01]:
        h = (2 * np.pi) / n
        # tau_func = lambda alpha, h:

        tau = (alpha * h ** 2) / 4  # * 2*np.pi #/(2*np.pi), endret på a matrisen!!
        m = int(T / tau)

        U_num, Uex, Udiff = heat_equation_solver_manufactured_solution(u_func, g, kappa=1, theta=theta, n=n, a=a, b=b,
                                                                       tau=tau, t0=t0, T=T, homogeneous=True,
                                                                       u_0=u_0_step)
        x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]

        ani = plot_2D_animation(x, y, U_num, title=r"$\alpha = $" + str(alpha), duration=10, zlim=(-1, 1))
        plt.show()
