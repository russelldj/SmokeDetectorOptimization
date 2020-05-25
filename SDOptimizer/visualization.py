import pdb
from mpl_toolkits.mplot3d import Axes3D
from SDOptimizer.functions import normalize
from SDOptimizer.constants import PLOT_TITLES, BIG_NUMBER, ALARM_THRESHOLD, PAPER_READY
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')


def show_optimization_statistics(vals, iterations, locs):
    plt.hist(vals, bins=10)  # plot the bins as a quarter of the spread
    # ax1.violinplot(vals)
    plt.xlabel("objective function values")
    if PAPER_READY:
        plt.savefig("vis/SummaryOptimizationValues.png")
    plt.show()

    plt.hist(iterations, bins=10)
    # TODO make this a histogram
    plt.xlabel("number of evaluations to converge")
    if PAPER_READY:
        plt.savefig("vis/SummaryOptimizationIterations.png")
    plt.show()

    for loc in locs:
        plot_xy(loc)
    plt.xlim(0, 8.1)
    plt.ylim(0, 3)
    plt.show()


def show_optimization_runs(all_funcs_values):
    """
    show the distribution of training curves

    all_funcs_values : ArrayLike[ArrayLike]
        the objective function value versus iteration for all the runs
    """
    for func_values in all_funcs_values:
        plt.plot(func_values)
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective function value")
    if PAPER_READY:
        plt.savefig("vis/ObjectiveFunctionValues.png")
    plt.title("The plot of all the objective functions for a set of runs")
    plt.show()

    # plot the summary statistics


def plot_xy(xy):
    plt.scatter(xy[::2], xy[1::2])


def plot_sphere(phi, theta, cs, r=1):
    phi = normalize(phi, -np.pi, 2 * np.pi)
    theta = normalize(theta, -np.pi, 2 * np.pi)
    xs = r * np.sin(phi) * np.cos(theta)
    ys = r * np.sin(phi) * np.sin(theta)
    zs = r * np.cos(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.savefig("sphere.png")
    pdb.set_trace()
