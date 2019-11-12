import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from project.diff_2d import fdm_poisson_2d_matrix_dense, apply_bcs, I, plot2D
from project.utils import plot_2D_animation

a, b = 0, 1
n = 10
h = (b - a) / n

N = (n + 1) ** 2

m = 10  # Time steps
t0 = 0  # sek
T = 1  # sek


f = lambda x, y, t: np.ones((len(x), len(x)))

x, y = np.ogrid[0:1:(n + 1) * 1j, 0:1:(n + 1) * 1j]

A = fdm_poisson_2d_matrix_dense(n, I)
Id = np.eye(A.shape[0])

tau = 0.1
theta = 1


def u_func(x, y, t, pck=np):
    k, l = pck.pi, 2*pck.pi
    mu = 1
    return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)


u_field = u_func(x, y, 0)
U_k1 = np.array([u_field[i, j] for j in range(n + 1) for i in range(n + 1)]).reshape((-1, 1))

U_0_field = U_k1.reshape((n + 1, n + 1))


F_k1 = f(x, y, 0).ravel().reshape((-1, 1))

Us = [U_0_field]

for k in range(m):
    U_k = U_k1
    t_k = k * (T - t0) / m
    t_k1 = (k + 1) * (T - t0) / m

    F_k = F_k1
    F_k1 = f(x, y, t_k1).ravel().reshape((-1, 1))

    B_k1 = (Id - tau * (1 - theta) * A) @ U_k + tau * theta * F_k1 + tau * (1 - theta) * F_k

    # TODO boundary conditions: modify B_k1 for this

    U_k1 = np.linalg.solve((Id - tau * theta * A), B_k1)
    U_k1_field = U_k1.reshape((n + 1, n + 1))

    Us.append(U_k1_field)
    plot2D(x, y, U_k1_field, "$U_" + str(k + 1) + "$")


# plot_2D_animation(x, y, Us, title="Us", duration=10, zlim=(-1, 1))
# plt.show()
