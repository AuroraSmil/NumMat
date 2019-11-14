import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse as sc

try:
    from project.diff_2d import apply_bcs, I, plot2D
    from project.utils import plot_2D_animation
except ModuleNotFoundError:
    from diff_2d import apply_bcs, I, plot2D
    from utils import plot_2D_animation


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


def laplace_u(u_func, x, y):
    # Automatic differentiation of u_func with sympy.
    dell_x = sp.diff(u_func, x)
    dell_y = sp.diff(u_func, y)
    # Set MINUS in front, to match diff equation
    laplace = sp.diff(dell_x, x) + sp.diff(dell_y, y)
    return laplace


def f_expression(u_func, kappa):
    x_var, y_var, t_var = sp.var("x_var y_var t_var")
    u_func_sp = u_func(x_var, y_var, t_var, sp)
    f_exp = sp.diff(u_func_sp, t_var) - kappa * laplace_u(u_func_sp, x_var, y_var)
    print("f edp", f_exp)
    return sp.lambdify((x_var, y_var, t_var), f_exp, "numpy")


def heat_equation_solver_manufactured_solution(u_func, g, kappa, theta, n, a, b, m, t0, T, homogeneous=False):
    # Create x, y grid
    x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]
    # Time step
    h = (b - a) / n
    N = (n + 1) ** 2
    tau = (T - t0) / m
    A = fdm_poisson_2d_matrix_sparse(n, I)
    Id = np.eye(A.shape[0])

    if homogeneous:
        def f(x, y, t):
            return x * y * 0
    else:
        f = f_expression(u_func, kappa)

    ####################################

    u_field = u_func(x, y, t0)
    # U_0
    U_k1 = np.array([u_field[i, j] for j in range(n + 1) for i in range(n + 1)]).reshape((-1, 1))
    U_0_field = U_k1.reshape((n + 1, n + 1))

    # F_0
    F_k1 = f(x, y, 0).ravel().reshape((-1, 1))

    Us = [U_0_field]
    U_diff = [U_0_field]
    U_exact = [U_0_field]

    for k in range(m):
        U_k = U_k1
        t_k = k * tau
        t_k1 = (k + 1) * tau

        F_k = F_k1
        F_k1 = f(x, y, t_k1).ravel().reshape((-1, 1))

        # Calculate new B_k1
        B_k1 = (Id - tau * (1 - theta) * A) @ U_k + tau * theta * F_k1 + tau * (1 - theta) * F_k

        # Apply border conditions
        G = g(x, y, t_k).reshape((-1, 1))
        B_k1 = apply_bcs(B_k1, G, n, I)

        # Solve linear system
        U_k1 = np.linalg.solve((Id + tau * theta * A), B_k1)

        U_k1_field = U_k1.reshape((n + 1, n + 1))  # Numerical solution as n*n field
        u_field = u_func(x, y, t_k)  # Exact solution

        Us.append(U_k1_field)
        U_exact.append(u_field)
        U_diff.append(np.abs(U_k1_field - u_field))

    return Us, U_exact, U_diff


def main():
    # Grid size
    a, b = 0, 2 * np.pi
    n = 10

    m = n  # Time steps, skal være lit n
    t0 = 0  # sek t start
    T = 1  # sek t stlutt

    theta = 1

    kappa = 1.1

    # Exact function
    def u_func(x, y, t, pck=np):
        k, l = 1, 1
        mu = k ** 2 + l ** 2
        return pck.sin(k * x) * pck.sin(l * y) * pck.exp(-mu * t)

    g = u_func

    x, y = np.ogrid[a:b:(n + 1) * 1j, a:b:(n + 1) * 1j]
    _, _, U_diff = heat_equation_solver_manufactured_solution(u_func, g, kappa, theta, n, a, b, m, t0, T,
                                                              homogeneous=False)
    ani = plot_2D_animation(x, y, U_diff, title="Us", duration=10, zlim=(-1, 1))
    # ani = plot_2D_animation(x, y, U_diff, title="Us", duration=10, zlim=(-1, 1))
    plt.show()


if __name__ == '__main__':
    main()
