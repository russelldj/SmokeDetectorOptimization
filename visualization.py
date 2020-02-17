import numpy as np
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
from constants import *

def show_optimization_statistics(vals, iterations, locs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.hist(vals, bins=10, rwidth=0.25) # plot the bins as a quarter of the spread
    ax1.violinplot(vals)
    ax1.set_ylabel("objective function values") # TODO make this a histogram
    #ax2.hist(iterations, bins=10)
    ax2.violinplot(iterations)
    ax2.set_ylabel("number of iterations function values") # TODO make this a histogram
    plt.show()
    for loc in locs:
        plot_xy(loc)
    plt.xlim(0, 8.1)
    plt.ylim(0, 3)
    plt.show()


def plot_xy(xy):
    plt.scatter(xy[::2], xy[1::2])
