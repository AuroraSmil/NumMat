####### task 5
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse as sc

from full_heat_eqn import fdm_poisson_2d_matrix_sparse, u_func
from diff_2d import apply_bcs, I, plot2D
from utils import plot_2D_animation

a, b = 0, 2*np.pi
n = 20
h = (b - a) / n
N = (n + 1) ** 2

t0 = 0  # sek t start
T = 1  # sek t stlutt
theta  = 0

alpha = [0.9, 1, 1.1]
tau_func = lambda alpha, h: alpha*h**2/4

tau = tau_func(alpha[2], h)
print(tau)
m = int(2*np.pi/tau)  # Time steps, skal være lit n

print(m)

#lager griddet
x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]

A = fdm_poisson_2d_matrix_sparse(n, I)
Id = np.eye(A.shape[0])


def u_0_step(x, y):
    step = 0*np.ones_like(x)*np.ones_like(y)

    def is_inside(x, y):
        eps = 1e-10
        return  np.abs(x - np.pi) < np.pi/3  and np.abs(y - np.pi) < np.pi/3

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            if is_inside(x[i,0], y[0,j]):
                step[i,j] = 1
    return step


#f = f_expression()
#g = u_func

u_field = u_0_step(x,y)
U_0_field = u_field
U_k1 = U_0_field.ravel().reshape((-1, 1))

Us = [U_0_field]

for k in range(m):
    U_k = U_k1
    t_k = k * tau
    t_k1 = (k + 1) * tau


    #print(np.shape(Id))
    #print(np.shape(A))
    #print(np.shape(U_k))
    B_k1 = (Id - tau * (1 - theta) * A) @ U_k #+ tau * theta * F_k1 + tau * (1 - theta) * F_k hvis F = 0

    #boundary conditions
    for j in [0, n]:
        for i in range(n + 1):
            B_k1[I(i, j, n)] =0 # g(a + i * h, a + j * h, t_k)

    for i in [0, n]:
        for j in range(n + 1):
            B_k1[I(i, j, n)] = 0 #g(a + i * h, a + j * h, t_k)

    a =1
    U_k1 = np.linalg.solve((Id + tau * theta * A), B_k1)

    U_k1_field = U_k1.reshape((n + 1, n + 1))

    Us.append(U_k1_field)
    u_field = u_func(x, y, t_k)
    # U_0
    U_ex = np.array([u_field[i, j] for j in range(n + 1) for i in range(n + 1)]).reshape((-1, 1))


    U_ex_field = U_ex.reshape((n + 1, n + 1))

    #U_exakt.append(U_k1_field - U_ex_field)
    #plot2D(x, y, U_k1_field, "$U_" + str(k + 1) + "$")

ani = plot_2D_animation(x, y, Us, title="Uk", duration=10, zlim=(-1, 1))
plt.show()
