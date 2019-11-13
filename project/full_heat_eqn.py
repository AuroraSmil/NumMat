import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse as sc

from project.diff_2d import apply_bcs, I, plot2D
from project.utils import plot_2D_animation


def fdm_poisson_2d_matrix_sparse(n, I):
    # Grid size
    h = 1.0 / n

    # Total number of unknowns is N = (n+1)*(n+1)
    N = (n + 1) ** 2

    # Define zero matrix A of right size and insert 0
    A = sc.dok_matrix((N, N))

    # Define FD entries of A
    hh = h * h
    for j in range(1, n):
        for i in range(1, n):
            A[I(i, j, n), I(i, j, n)] = 4 / hh  # U_ij, center
            A[I(i, j, n), I(i - 1, j, n)] = -1 / hh  # U_{i-1,j}, left
            A[I(i, j, n), I(i + 1, j, n)] = -1 / hh  # U_{i+1,j}, right
            A[I(i, j, n), I(i, j - 1, n)] = -1 / hh  # U_{i,j-1}, under
            A[I(i, j, n), I(i, j + 1, n)] = -1 / hh  # U_{i,j+1}, over

    A_csr = A.tocsr()
    return A_csr

#Angir initialverdier
a, b = 0, 2*np.pi
n = 10

h = (b - a) / n
N = (n + 1) ** 2

m = n  # Time steps, skal være lit n
t0 = 0.5  # sek t start
T = 1  # sek t stlutt

#lager griddet
x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]


A = fdm_poisson_2d_matrix_sparse(n, I)
Id = np.eye(A.shape[0])

#timestep
tau = (T-t0) / m
theta = 0.5

k, l = 1, 1 #HVA SKAL DISSE VÆRE!!!!!
mu = k ** 2 + l ** 2
kappa = 1.1 # HVA SKAL DENNE VÆRE

#Ekstakt funksjon
def u_func(x, y, t, pck=np):
    return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)


def laplace_u(u_func, x, y):
    # Automatic differerentiation of u_func with sympy.
    dell_x = sp.diff(u_func, x)
    dell_y = sp.diff(u_func, y)
    # Set MINUS in front, to match diff equation
    laplace = sp.diff(dell_x, x) + sp.diff(dell_y, y)
    return laplace


def f_expression():
    x_var, y_var, t_var = sp.var("x_var y_var t_var")
    u_func_sp = u_func(x_var, y_var, t_var, sp)
    f_exp = sp.diff(u_func_sp, t_var) - kappa * laplace_u(u_func_sp, x_var, y_var)
    print("f edp", f_exp)
    return sp.lambdify((x_var, y_var, t_var), f_exp, "numpy")


f = f_expression()


"""
x_v, y_v, t_v = sp.var("x_v y_v t_v")

res = u_func(x_v, y_v, t_v, sp)

print(res)
print(laplace_u(res, x_v, y_v))
print("")
print(f(1, 2, 3))


exit()
"""

g = u_func

u_field = u_func(x, y, 0)
# U_0
U_k1 = np.array([u_field[i, j] for j in range(n + 1) for i in range(n + 1)]).reshape((-1, 1))
U_0_field = U_k1.reshape((n + 1, n + 1))


# F_0
F_k1 = f(x, y, 0).ravel().reshape((-1, 1))

Us = [U_0_field]

for k in range(m):
    U_k = U_k1
    t_k = k * tau # endret fra (T-t0) / m til tau
    t_k1 = (k + 1) * tau

    F_k = F_k1
    F_k1 = f(x, y, t_k1).ravel().reshape((-1, 1))

    B_k1 = (Id - tau * (1 - theta) * A) @ U_k + tau * theta * F_k1 + tau * (1 - theta) * F_k

    #boundary conditions
    for j in [0, n]:
        for i in range(n + 1):
            B_k1[I(i, j, n)] = g(a + i * h, a + j * h, t_k)

    for i in [0, n]:
        for j in range(n + 1):
            B_k1[I(i, j, n)] = g(a + i * h, a + j * h, t_k)

    U_k1 = np.linalg.solve((Id - tau * theta * A), B_k1)
    U_k1_field = U_k1.reshape((n + 1, n + 1))

    Us.append(U_k1_field)
    plot2D(x, y, U_k1_field, "$U_" + str({k + 1}) + "$")

print(Us)
plot_2D_animation(x, y, Us, title="Us", duration=10, zlim=(-1, 1)) #U er liste av matricer ikke array!!
plt.show()
