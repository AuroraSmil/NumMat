import pickle

import numpy as np

try:
    from project.full_heat_eqn import heat_equation_solver_manufactured_solution
    from project.utils import plot_2D_animation
except Exception:
    from full_heat_eqn import heat_equation_solver_manufactured_solution
    from utils import plot_2D_animation


# MÅ FÅ DEN TIL Å FUNKE FOR N=80 OGSÅ :)
def main():
    # Grid size
    a, b = 0, 2 * np.pi
    t0 = 0  # sek t start
    T = 1  # sek t stlutt
    kappa = 1.1

    # Function
    def u_func(x, y, t, pck=np):
        k, l = 1, 1
        mu = k ** 2 + l ** 2
        return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)

    g = u_func

    exact_solutions = []
    numerical_solutions = []
    diffs = []

    # Loop through choices.
    for theta in [0, 0.5, 1]:
        for n in [10, 20, 40]:  # , 80]:
            print("theta", theta, "n", n)
            h = 2 * np.pi / n
            tau = h / (2 * np.pi)
            print("tau", tau)

            u_num, u_ex, u_diff = heat_equation_solver_manufactured_solution(u_func, g, kappa, theta=theta, n=n, a=a,
                                                                             b=b, tau=tau, t0=t0, T=T, homogeneous=True)

            meta = {"theta": theta, "n": n}
            exact_solutions.append((meta, u_ex))
            numerical_solutions.append((meta, u_num))
            diffs.append((meta, u_diff))
            """
            """

    return numerical_solutions, exact_solutions, diffs


if __name__ == '__main__':
    all_numerical, all_exact, all_diff = main()
    with open("numerical", "wb") as f:
        pickle.dump([all_numerical, all_exact, all_diff], f)

    """
    a, b = 0, 2*np.pi

    for num, ex, diff in zip(all_numerical, all_exact, all_diff):

        n = num[0]["n"]
        x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]

        ani = plot_2D_animation(x, y, num[1], title="numerical" + str(num[0]), duration=1, zlim=(-1, 1))
        ani2 = plot_2D_animation(x, y, ex[1], title="exact" + str(ex[0]), duration=1, zlim=(-1, 1))
        ani3 = plot_2D_animation(x, y, diff[1], title="diff" + str(diff[0]), duration=1, zlim=(-1, 1))

        plt.show()

    """
