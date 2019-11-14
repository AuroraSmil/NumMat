####### task 5
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse as sc

from full_heat_eqn import heat_equation_solver_manufactured_solution
from diff_2d import apply_bcs, I, plot2D
from utils import plot_2D_animation



if __name__ == '__main__':
    print("hei")
    a, b = 0, 2*np.pi
    n = 20


    N = (n + 1) ** 2
    t0 = 0  # sek t start
    T = 20  # sek t stlutt
    theta  = 0

    alpha = [0.9, 1, 1.01]

    h = (2 * np.pi) / n
    tau_func = lambda alpha, h: (alpha*h**2)/4

    tau = tau_func(alpha[2], h) #* 2*np.pi #/(2*np.pi), endret på a matrisen!!
    print(tau)
    m = int(T/tau)


    # Exact function
    def u_func(x, y, t, pck=np):
        k, l = 1, 1
        mu = k ** 2 + l ** 2
        return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)

    g = u_func
    def g(x, y, t):
        return x * y *0

    U_num, Uex, Udiff = heat_equation_solver_manufactured_solution(u_func, g, kappa=1, theta=theta, n=n, a=a, b=b, m=m, t0=t0, T=T, homogeneous=True)
    x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]

    ani = plot_2D_animation(x, y, U_num, title="Us", duration=10, zlim=(-1, 1))
    # ani = plot_2D_animation(x, y, U_diff, title="Us", duration=10, zlim=(-1, 1))
    plt.show()

