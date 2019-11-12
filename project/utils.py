import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np


def _plot_frame_fdm_solution(i, ax, X, Y, U_list, title, zlim=None):
    ax.clear()
    line = ax.plot_surface(X, Y, U_list[i],
                           rstride=1, cstride=1,  # Sampling rates for the x and y input data
                           cmap=cm.viridis)  # Use the new fancy colormap viridis
    if zlim is not None:
        ax.set_zlim(zlim)
    total_frame_number = len(U_list)
    complete_title = title + (" (Frame %d of %d)" % (i, total_frame_number))
    ax.set_title(complete_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return line


def plot_2D_animation(X, Y, U_list, title='', duration=10, zlim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fargs = (ax, X, Y, U_list, title, zlim)
    frame_plotter = _plot_frame_fdm_solution

    frames = len(U_list)
    interval = duration / frames * 1000
    ani = animation.FuncAnimation(fig, frame_plotter,
                                  frames=len(U_list), fargs=fargs,
                                  interval=interval, blit=False, repeat=True)
    return ani


if __name__ == '__main__':

    # Define a time-dependent function
    def u_ex(x, y, t):
        return np.exp(-2 * t) * np.sin(x) * np.sin(y)


    t, T = 0, 1
    tau = 0.01

    # Generate grid
    L = 2 * np.pi
    n = 10
    xi = np.linspace(0, L, n + 1)
    yi = np.linspace(0, L, n + 1)
    X, Y = np.meshgrid(xi, yi, sparse=True)

    # Store U in a list for animation plot
    U_list = [u_ex(X, Y, 0)]

    # Evaluate exact solution at each time step and store it
    while t < T:
        t += tau
        U = u_ex(X, Y, t)
        U_list.append(U)

    # Set lower and upper z axis limit to avoid rescaling during simulation
    zlim = (-1.0, 1.0)
    # Create animation
    ani = plot_2D_animation(X, Y, U_list, title='', duration=10, zlim=zlim)
    plt.show()
