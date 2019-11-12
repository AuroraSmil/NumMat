import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Define index mapping
def I(i, j, n):
    return i + j * (n + 1)


#########################################################################################################

def fdm_poisson_2d_matrix_dense(n, I):
    # Grid size
    h = 1.0 / n

    # Total number of unknowns is N = (n+1)*(n+1)
    N = (n + 1) ** 2

    # Define zero matrix A of right size and insert 0
    A = np.zeros((N, N))

    # Define FD entries of A
    hh = h * h
    for j in range(1, n):
        for i in range(1, n):
            A[I(i, j, n), I(i, j, n)] = 4 / hh  # U_ij, center
            A[I(i, j, n), I(i - 1, j, n)] = -1 / hh  # U_{i-1,j}, left
            A[I(i, j, n), I(i + 1, j, n)] = -1 / hh  # U_{i+1,j}, right
            A[I(i, j, n), I(i, j - 1, n)] = -1 / hh  # U_{i,j-1}, under
            A[I(i, j, n), I(i, j + 1, n)] = -1 / hh  # U_{i,j+1}, over

    # Incorporate boundary conditions
    # Add points to grid related to boundary values on the bottom and on the top.
    for j in [0, n]:
        for i in range(0, n + 1):
            # print("indeks", I(i, j, n))
            A[I(i, j, n), I(i, j, n)] = 1

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, n]:
        for j in range(0, n + 1):
            # print("ind", I(i, j, n))
            A[I(i, j, n), I(i, j, n)] = 1

    return A


# A = fdm_poisson_2d_matrix_dense(3, I)


#########################################################################################################


# Number of subdivisions in each dimension
n = 50

# To define the grid we could use "linspace" as
# in the first part to define subdivisions for
# the $x$ and $y$ axes. But to make plotting easy
# and to vectorize the evaluation of the right-hand
# side $f$, we do something more fancy. We define
# x and y coordinates for the grid using a
# "sparse grid" representation using the function 'ogrid'.
# (Read the documentation for 'ogrid'!).
# Unfortunately, ogrid does not include the interval
# endpoints by
# default, but according to the numpy documentation,
# you can achieve this by multiplying your sampling number by
# the pure imaginary number $i = \sqrt{-1}$  which is written as "1j" in Python code.
# So simply speaking "(N+1)*1j" reads "include the end points"
# while (N+1) reads "exclude the end points".
x, y = np.ogrid[0:1:(n + 1) * 1j, 0:1:(n + 1) * 1j]

# Print x and y to see how they look like!
# print(x)
# print(y)


#########################################################################################################


x_var, y_var = sp.var("x_var y_var")


# Example of exact solution
def u_ex(x, y, pck=np):
    return pck.sin(1 * pck.pi * x) * pck.sin(2 * pck.pi * y)


def laplace_u(x, y):
    # Automatic differerentiation of u_ex with sympy.
    u_sp = u_ex(x_var, y_var, sp)
    dell_x = sp.diff(u_sp, x_var)
    dell_y = sp.diff(u_sp, y_var)
    # Set MINUS in front, to match diff equation
    laplace = -(sp.diff(dell_x, x_var) + sp.diff(dell_y, y_var))
    # Return as numpy function
    return sp.lambdify((x_var, y_var), laplace, "numpy")(x, y)


# Boundary data g is given by u_ex
g = u_ex


# Right hand side
def f(x, y):
    return laplace_u(x, y)


# Evaluate u on the grid. The output will be a 2 dimensional array
# where U_ex_grid[i,j] = u_ex(x_i, y_j)
U_ex_grid = u_ex(x, y)


#########################################################################################################


def plot2D(X, Y, Z, title=""):
    # Define a new figure with given size an
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1,  # Sampling rates for the x and y input data
                           cmap=cm.viridis)  # Use the new fancy colormap viridis

    # Set initial view angle
    ax.view_init(30, 225)

    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()


#########################################################################################################
if __name__ == '__main__':

    plot2D(x, y, U_ex_grid, title="$u(x,y)$")
    plt.show()

    #########################################################################################################

    # Evaluate f on the grid. The output will be a 2 dimensional array
    # where f_grid[i,j] = f(x_i, y_j)
    F_grid = f(x, y)

    # Same game for boundary data g
    G_grid = g(x, y)

    #########################################################################################################


    # To apply bcs we have to flatten out F which is done by the ravel function
    F = F_grid.ravel()

    # To apply bcs we have to flatten out G which is done by the ravel function
    G = G_grid.ravel()


    #########################################################################################################


def apply_bcs(F, G, n, I):
    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [I(i, j, n) for j in [0, n] for i in range(0, n + 1)] + \
                 [I(i, j, n) for i in [0, n] for j in range(0, n + 1)]
    F[bc_indices] = G[bc_indices]

    return F


#########################################################################################################


if __name__ == '__main__':

    # Linear algebra solvers from scipy
    import scipy.linalg as la

    # Compute the FDM matrix
    A = fdm_poisson_2d_matrix_dense(n, I)

    # Apply bcs
    F = apply_bcs(F, G, n, I)

    print("SOLVING")
    print(A.shape)
    print(F.shape)

    # Solve
    U = la.solve(A, F)

    # Make U into a grid function for plotting
    U_grid = U.reshape((n + 1, n + 1))

    print("SOLVED")

    # and plot f
    plot2D(x, y, U_grid, title="$u(x,y) løst$")
    plt.show()


    ######

def fdm_poisson_2d_matrix_sparse(n, I):
    # Grid size
    h = 1.0 / n

    # Total number of unknowns is N = (n+1)*(n+1)
    N = (n + 1) ** 2

    # Define zero matrix A of right size and insert 0
    A = sp.dok_matrix((N, N))

    # Define FD entries of A
    hh = h * h
    for j in range(1, n):
        for i in range(1, n):
            A[I(i, j, n), I(i, j, n)] = 4 / hh  # U_ij, center
            A[I(i, j, n), I(i - 1, j, n)] = -1 / hh  # U_{i-1,j}, left
            A[I(i, j, n), I(i + 1, j, n)] = -1 / hh  # U_{i+1,j}, right
            A[I(i, j, n), I(i, j - 1, n)] = -1 / hh  # U_{i,j-1}, under
            A[I(i, j, n), I(i, j + 1, n)] = -1 / hh  # U_{i,j+1}, over

    # Incorporate boundary conditions
    # Add points to grid related to boundary values on the bottom and on the top.
    for j in [0, n]:
        for i in range(0, n + 1):
            # print("indeks", I(i, j, n))
            A[I(i, j, n), I(i, j, n)] = 1

    # Add boundary values related to unknowns from the first and last grid COLUMN
    for i in [0, n]:
        for j in range(0, n + 1):
            # print("ind", I(i, j, n))
            A[I(i, j, n), I(i, j, n)] = 1
    A_csr = A.tocsr()
    return A_csr


    plot2D(x, y, U_ex_grid - U_grid, title="difference")
    plt.show()
