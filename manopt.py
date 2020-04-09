from SDOptimizer.functions import make_total_lookup_function

import autograd.numpy as np

from pymanopt.manifolds import Stiefel, Sphere
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, NelderMead
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Creating the data:
# I'm going to create a cuboid of sampled points on the regions (-1, 1) x (-1, 1) x (-1, 1)
# then compute a cost function for each of them
# then map this points into phi, rho spaces
# and keep the original associated cost function
# now do the optimization over the sphere in R3
# for each point on the sphere, convert it into phi, rho
# do the lookup on the phi, rho grid and report that value

# (1) Instantiate a manifold
manifold = Sphere(3)

def create_square(x_bounds, y_bounds, z_bounds, samples=50):
    """
    each of the first arguements should be in the form [lower, upper] except for on in the form [value]
    e.g. create_square([-1, 1], [-1, 1], [1])

    returns (n, 3) matrix representing a face
    """
    lengths = np.array([len(x_bounds), len(y_bounds), len(z_bounds)])
    if np.array_equal(lengths, np.array([1, 2, 2])):
        xs = np.ones((samples, samples)) * x_bounds[0]

        ys = np.linspace(y_bounds[0], y_bounds[1], num=samples)
        zs = np.linspace(z_bounds[0], z_bounds[1], num=samples)
        ys, zs = np.meshgrid(ys, zs)
    elif np.array_equal(lengths, np.array([2, 1, 2])):
        ys = np.ones((samples, samples)) * y_bounds[0]

        xs = np.linspace(x_bounds[0], x_bounds[1], num=samples)
        zs = np.linspace(z_bounds[0], z_bounds[1], num=samples)
        xs, zs = np.meshgrid(xs, zs)
    elif np.array_equal(lengths, np.array([2, 2, 1])):
        zs = np.ones((samples, samples)) * z_bounds[0]

        xs = np.linspace(x_bounds[0], x_bounds[1], num=samples)
        ys = np.linspace(y_bounds[0], y_bounds[1], num=samples)
        xs, ys = np.meshgrid(xs, ys)
    else:
        raise ValueError("The is something wrong with the bounds, two should be length 2 and one should be length 1")

    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    all = np.vstack((xs, ys, zs))
    return all.transpose()

def create_cube(x_bounds, y_bounds, z_bounds, samples=50):
    """
    all the lengths of the bounds should be 2
    e.g create_cube([-1, 1], [-1, 1], [-1, 1])

    returns a (3, 6 * samples^2) array where each row represents a dimension
    """
    all_faces = np.zeros((0, 3))

    x_low, x_high = x_bounds
    y_low, y_high = y_bounds
    z_low, z_high = z_bounds
    COMBINATIONS = [
        [x_bounds, y_bounds, [z_low]],
        [x_bounds, y_bounds, [z_high]],
        [x_bounds, [y_low], z_bounds],
        [x_bounds, [y_high], z_bounds],
        [[x_low], y_bounds, z_bounds],
        [[x_high], y_bounds, z_bounds]
        ]

    for comb in COMBINATIONS:
        face = create_square(*comb)
        all_faces = np.vstack((all_faces, face))

    return all_faces

def xyz_to_spherical(xyz):
    """
    modified from
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    r_elev_ax = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r_elev_ax[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    r_elev_ax[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    r_elev_ax[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return r_elev_ax

def spherical_to_xyz(elev_az):
    phi = elev_az[:,0] # check that these aren't switched and migrate to all one convention
    theta = elev_az[:,1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    xyz = np.vstack((x, y, z))
    return xyz.transpose()


def make_cost_func(is_manifold=False):
    """
    is_manifold : Boolean
        are we optimizing over xyz points constrained to lie on a sphere embedded in R3 or
        the elev, azimuthal coordnates in R2
    """
    BOUNDS = [[-1, 1], [-1, 1], [-1, 1]]
    all_faces = create_cube(*BOUNDS)
    cost_per_point = np.sum(all_faces, axis=1)
    r_elev_ax = xyz_to_spherical(all_faces)
    new_xyz = spherical_to_xyz(r_elev_ax[:, 1:])
    print("xyz mins : {}".format(np.min(all_faces, axis=0)))
    print("xyz maxes : {}".format(np.max(all_faces, axis=0)))
    print(np.max(cost_per_point))

    print("mins : {}".format(np.min(r_elev_ax, axis=0)))
    print("maxes : {}".format(np.max(r_elev_ax, axis=0)))
    print(r_elev_ax.shape)
    elev = r_elev_ax[:,1]
    ax = r_elev_ax[:,2]
    cost_func_elev_az = make_total_lookup_function([(elev, ax, cost_per_point)]) # maps from a elev, ax to a cost
    plt.scatter(elev, ax, c=cost_per_point)
    plt.pause(1)
    if not is_manifold:
        return cost_func_elev_az

    def cost_fun_xyz(xyz):
        if len(xyz.shape) == 1:
            xyz = np.expand_dims(xyz, axis=0)
        r_elev_ax = xyz_to_spherical(xyz)
        elev_ax = r_elev_ax[0, 1:]  # get the ()
        cost = cost_func_elev_az(elev_ax)
        return cost

    return cost_fun_xyz


cost_fun = make_cost_func()
point = np.array([[1, 1, 1]])
print(point.shape)
r_elev_ax = xyz_to_spherical(point)
elev_ax = r_elev_ax[0, 1:]
print(cost_fun(np.array([0, 1])))
print(cost_fun(np.array([np.pi / 2.0, np.pi / 4.0])))
print(cost_fun(elev_ax))

BOUNDS = [[-1, 1], [-1, 1], [-1, 1]]
all_faces = create_cube(*BOUNDS)
cost_per_point = np.sum(all_faces, axis=1)
r_elev_ax = xyz_to_spherical(all_faces)
new_xyz = spherical_to_xyz(r_elev_ax[:, 1:])

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2], c=cost_per_point)
fig.show()
pdb.set_trace()


# (2) Define the cost function (here using autograd.numpy)
def cost(X): return np.sum(X)


problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
solver = NelderMead()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)
