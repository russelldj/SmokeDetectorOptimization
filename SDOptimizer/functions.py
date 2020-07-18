import numpy as np
from scipy.interpolate import griddata
import pdb
from SDOptimizer.constants import INTERPOLATION_METHOD
from SDOptimizer.constants import *


def make_location_objective(masked):
    """
    an example function to evalute the quality of the locations
    """
    if masked:
        def location_evaluation(xyons):  # TODO make this cleaner
            good = []
            for i in range(0, len(xyons), 3):
                x, y, on = xyons[i:i+3]
                if on[0]:
                    good.extend([x, y])
            if len(good) == 0:  # This means none were on
                return 0
            else:
                return np.linalg.norm(good)
    else:
        def location_evaluation(xys):
            return np.linalg.norm(xys)
    return location_evaluation


def make_counting_objective():
    """
    count the number of sources which are turned on
    """
    def counting_func(xyons):
        val = 0
        for i in range(0, len(xyons), 3):
            x, y, on = xyons[i:i+3]
            if on[0]:
                val += 1
        return val
    return counting_func


def make_lookup(X, Y, time_to_alarm, Z=None, interpolation_method=INTERPOLATION_METHOD):
    """
    X, Y : ArrayLike[Float]
        The x, y locations of the data points from the simulation
    time_to_alarm : ArrayLike[Float]
        The time to alarm corresponding to each of the locations
    Z : ArrayLike[Float]
        This is optional but will be used if not None
    interpolation_method : str
        The method for interpolating the data
    -----returns-----
    The sampled value, either using the nearest point or interpolation
    """
    if Z is None:
        # combine the x and y data points
        simulated_points = np.vstack((X, Y)).transpose()
        num_dimensions = 2  # The number of dimensions we are optimizing over
    else:
        # combine the x and y data points
        simulated_points = np.vstack((X, Y, Z)).transpose()
        num_dimensions = 3  # The number of dimensions we are optimizing over

    def ret_func(query_point):  # this is what will be returned
        """
        query_point : ArrayLike[float]
            The point to get the value for
        """
        if len(query_point) != num_dimensions:
            raise ValueError("The number of dimensions of the query point was {} when it should have been {}".format(
                len(query_point), num_dimensions))

        interpolated_time = griddata(
            simulated_points, time_to_alarm, query_point,
            method=interpolation_method)
        return interpolated_time
    return ret_func


def make_total_lookup_function(
        xytimes,
        verbose=False,
        type="worst_case",
        masked=False,
        interpolation_method="nearest"):
    """
    This function creates and returns the function which will be optimized
    -----inputs------
    xytimes : ArrayLike[Tuple[ArrayLike[Float]]]
        A list of tuples in the form [(x, y, times), (x, y, times), ....] coresponding to the different smokes sources
    verbose : Boolean
        print information during functinon evaluations
    type : String
        What function to use, "worst_cast", "softened", "second"
    masked : bool
        Is the input going to be [x, y, on, x, y, on, ...] representing active detectors
    interpolation_method : str
        Can be either nearest or linear, cubic is acceptable but idk why you'd do that
        How to interpolate the sampled points
    -----returns-----
    ret_func : Function[ArrayLike[Float] -> Float]
        This is the function which will eventually be optimized and it maps from the smoke detector locations to the time to alarm
        A function mapping [x1, y1, x2, y2, ....], represnting the x, y coordinates of each detector,to the objective function value
    """
    # Create data which will be used inside of the function to be returned
    funcs = []
    for xytime in xytimes:
        # create all of the functions mapping from a location to a time
        # This is notationionally dense but I think it is worthwhile
        # We are creating a list of functions for each of the smoke sources
        # The make_lookup function does that
        if len(xytime) == 3:
            # This is if we're only optimizing over two location variables per detector
            X, Y, time = xytime
            funcs.append(make_lookup(X, Y, time,
                                     interpolation_method=interpolation_method))
        elif len(xytime) == 4:
            # Three location variables per detector
            X, Y, Z, time = xytime
            funcs.append(make_lookup(X, Y, time, Z=Z,
                                     interpolation_method=interpolation_method))

    def ret_func(xys):
        """
        xys : ArrayLike
            Represents the x, y location of each of the smoke detectors as [x1, y1, x2, y2]
            could also be the [x, y, on, x, y, on,...] but masked should be specified in make_total_lookup_function
        -----returns-----
        worst_source : Float
            The objective function for the input
        """
        all_times = []  # each internal list coresponds to a smoke detector location
        if masked:
            some_on = False
            for i in range(0, len(xys), 3):
                x, y, on = xys[i:i+3]
                if on[0]:  # don't evaluate a detector which isn't on, on is really a list of length 1
                    all_times.append([])
                    some_on = True
                    for func in funcs:
                        all_times[-1].append(func([x, y]))
            all_times = np.asarray(all_times)
            if not some_on:  # This means that no sources were turned on
                return BIG_NUMBER
        else:
            for i in range(0, len(xys), 2):
                all_times.append([])
                x, y = xys[i:i+2]
                for func in funcs:
                    all_times[-1].append(func([x, y]))
            all_times = np.asarray(all_times)
        if type == "worst_case":
            time_for_each_source = np.amin(all_times, axis=0)
            worst_source = np.amax(time_for_each_source)
            ret_val = worst_source
        elif type == "second":
            time_for_each_source = np.amin(all_times, axis=0)
            second_source = np.sort(time_for_each_source)[1]
            ret_val = second_source
        elif type == "softened":
            time_for_each_source = np.amin(all_times, axis=0)
            sorted = np.sort(time_for_each_source)[1]
            ALPHA = 0.3
            ret_val = (sorted[0] + ALPHA * sorted[1]) / (1 + ALPHA)
        elif type == "fastest":
            # print(all_times)
            # this just cares about the source-detector pair that alarms fastest
            ret_val = np.amin(all_times)
        else:
            raise ValueError("type is : {} which is not included".format(type))
        if verbose:
            print("all of the times are {}".format(all_times))
            print("The quickest detction for each source is {}".format(
                time_for_each_source))
            print(
                "The slowest-to-be-detected source takes {}".format(worst_source))
        return ret_val
    return ret_func


def convert_to_spherical_from_points(X, Y, Z):
    # Make each of them lie on the range (-1, 1)
    X = np.expand_dims(normalize(X, -1, 2), axis=1)
    Y = np.expand_dims(normalize(Y, -1, 2), axis=1)
    Z = np.expand_dims(normalize(Z, -1, 2), axis=1)
    xyz = np.concatenate((X, Y, Z), axis=1)
    elev_az = xyz_to_spherical(xyz)[:, 1:]
    return elev_az[:, 0], elev_az[:, 1]


def normalize(x, lower_bound=0, scale=1):
    if scale <= 0:
        raise ValueError("scale was less than or equal to 0")
    minimum = np.min(x)
    maximum = np.max(x)
    diff = maximum - minimum
    x_prime = (x - minimum) / diff
    x_prime = x_prime * scale + lower_bound
    return x_prime


def xyz_to_spherical(xyz):
    """
    xyz : np.array
        this is (n, 3) with one row for each x, y, z
    modified from
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    r_elev_ax = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r_elev_ax[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    # for elevation angle defined from Z-axis down
    r_elev_ax[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    # ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    r_elev_ax[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return r_elev_ax


def spherical_to_xyz(elev_az):
    """
    elev_az : np.array
        This is a (n, 2) array where the columns represent the elevation and the azimuthal angles
    """
    # check that these aren't switched and migrate to all one convention
    phi = elev_az[:, 0]
    theta = elev_az[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    xyz = np.vstack((x, y, z))
    return xyz.transpose()
