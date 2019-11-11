import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# Define index mapping
def I(i,j,n):
    return i + j * (n+1)


#########################################################################################################

def fdm_poisson_2d_matrix_dense(n, I):
    # Gridsize
    h = 1.0 / n

    # Total number of unknowns is N = (n+1)*(n+1)
    N = (n + 1) ** 2

    # Define zero matrix A of right size and insert 0
    A = np.zeros((N, N))

    # Define FD entries of A
    hh = h * h
    hh = 1
    for j in range(1, n):
        for i in range(1, n):
            print("Indeks", I(i, j, n))
            A[I(i, j, n), I(i, j, n)] = 4 / hh  # U_ij, center
            if i > 1:
                A[I(i, j, n), I(i - 1, j, n)] = -1 / hh  # U_{i-1,j}, left
            if i < n - 1:
                A[I(i, j, n), I(i + 1, j, n)] = -1 / hh  # U_{i+1,j}, right
            if j > 1:
                A[I(i, j, n), I(i, j - 1, n)] = -1 / hh  # U_{i,j-1}, under
            if j < n - 1:
                A[I(i, j, n), I(i, j + 1, n)] = -1 / hh  # U_{i,j+1}, over

    # Incorporate boundary conditions
    # Add boundary values related to unknowns from the first and last grid ROW
    # USIKKER HER!!
    """
    for j in [0, n]:
        for i in range(0, n + 1):
            A[I(i, j, n): j] = 1
    """

    """
    # Add boundary values related to unknowns from the first and last grid COLUMN
    # USIKKER HER!!
    for i in [0, n]:
        for j in range(0, n + 1):
            A[i: I(i, j, n)] = 1
    """

    return A


A = fdm_poisson_2d_matrix_dense(3, I)

print(A.shape)
print(A)
exit()


#########################################################################################################


# Number of subdivisions in each dimension
n = 10


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
x,y = np.ogrid[0:1:(n+1)*1j, 0:1:(n+1)*1j]
print(x)

print(y)
# Print x and y to see how they look like!
#print(x)
#print(y)


#########################################################################################################


x_var, y_var = sp.var("x_var y_var")


# Example of exact solution
def u_ex(x, y, pck=np):
    return pck.sin(1*pck.pi*x)*pck.sin(2*pck.pi*y)


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
U_ex_grid = u_ex(x,y)


# print(U_ex_grid)

"""
plt.imshow(U_ex_grid)
plt.xlabel("x")
plt.show()

lap = f(x, y)
print(lap.shape)
plt.imshow(lap)
plt.show()
"""

# Print f_grid  to see how it looks like!
# print(U_ex_grid)


#########################################################################################################

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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

plot2D(x, y, f(x, y), title="$u(x,y)$")
plt.show()


A = fdm_poisson_2d_matrix_dense(10, I)
print(A)
input()

#########################################################################################################

# Evaluate f on the grid. The output will be a 2 dimensional array
# where f_grid[i,j] = f(x_i, y_j)
F_grid = f(x,y)

# Same game for boundary data g
G_grid = g(x,y)


#########################################################################################################


# To apply bcs we have to flatten out F which is done by the ravel function
F = F_grid.ravel()

# To apply bcs we have to flatten out G which is done by the ravel function
G = G_grid.ravel()


#########################################################################################################


def apply_bcs(F, G, n, I):
    # Add boundary values related to unknowns from the first and last grid ROW
    bc_indices = [ I(i,j,n)  for j in [0, n] for i in range(0, n+1) ]
    F[bc_indices] = G[bc_indices]

    # Add boundary values related to unknowns from the first and last grid COLUMN
    bc_indices = ...
    ...

#########################################################################################################


# Linear algebra solvers from scipy
import scipy.linalg as la

# Compute the FDM matrix
...

# Apply bcs
...

# Solve
...

# Make U into a grid function for plotting
U_grid = U.reshape((n+1,n+1))

# and plot f
plot2D(x, y, U_grid, title="$u(x,y)$")