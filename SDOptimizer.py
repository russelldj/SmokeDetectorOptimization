#!/usr/bin/env python
# coding: utf-8

import pdb
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import scipy
import logging
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
from scipy.interpolate import griddata

DATA_FILE = "exportUSLab.csv"  # Points to the data Katie gave us
VISUALIZE = False
ALARM_THRESHOLD = 4e-20
BIG_NUMBER = 100
INFEASIBLE_VALUE = BIG_NUMBER * 1.5


class SDOptimizer():
    def __init__(self):
        self.DATA_FILE = DATA_FILE
        self.VISUALIZE = VISUALIZE
        self.ALARM_THRESHOLD = ALARM_THRESHOLD
        self.BIG_NUMBER = BIG_NUMBER
        self.logger = logging.getLogger('main')
        self.logger.debug("Instantiated the optimizer")

    def load_data(self, data_file):
        # This is a doubly-nested list with each internal list representing a
        # single timestep
        all_points = [[]]
        with open(data_file, 'r') as infile:  # read through line by line and parse by the time
            for line in infile:
                if "symmetry" in line:
                    all_points.append([line])
                all_points[-1].append(line)
        self.logger.info(
            "The number of timesteps is {}".format(len(all_points)))

        self.all_times = []
        self.max_consentrations = []

        for time in all_points[1:-1]:
            df = pd.read_csv(io.StringIO('\n'.join(time[4:])))
            df = df.rename(
                columns={
                    'Node Number': 'N',
                    ' X [ m ]': 'X',
                    ' Y [ m ]': 'Y',
                    ' Z [ m ]': 'Z',
                    ' Particle Mass Concentration [ kg m^-3 ]': 'C'})
            # drop the last row which is always none
            df.drop(df.tail(1).index, inplace=True)
            self.all_times.append(df)
            # get all of the consentrations but the null last one
            consentration = df['C'].values
            self.max_consentrations.append(np.max(consentration))

        # The x and y values are the same for all timesteps
        self.X = df['X'].values
        self.Y = df['Y'].values

    def visualize(self, show=False):
        print(self.max_consentrations)
        max_consentration = max(self.max_consentrations)
        print(max_consentration)

        for i, df in enumerate(self.all_times):
            plt.cla()
            plt.clf()
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.title("consentration at timestep {} versus position".format(i))
            norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
            cb = plt.scatter(
                df['X'],
                df['Y'],
                c=df['C'] /
                max_consentration,
                cmap=plt.cm.inferno,
                norm=norm)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.savefig("vis/consentration{:03d}.png".format(i))
            if show:
                plt.show()

    def get_time_to_alarm(self, flip_x=False, flip_y=False, infeasible_locations=None, alarm_threshold=ALARM_THRESHOLD, visualize=False):
        """The flips are just for data augmentation to create more example data
        infeasible_locations is an array of tuples where each one represents an object [(x1, y1, x2, y2),....]
        """
        consentrations = np.asarray(
            [x['C'].values for x in self.all_times])  # Get all of the consentrations
        self.logger.info("There are {} timesteps and {} flattened locations".format(
            consentrations.shape[0], consentrations.shape[1]))

        # Determine which entries have higher consentrations
        alarmed = consentrations > alarm_threshold
        nonzero = np.nonzero(alarmed)  # determine where the non zero entries
        # this is pairs indicating that it alarmed at that time and location
        nonzero_times, nonzero_locations = nonzero

        time_to_alarm = []
        for loc in range(alarmed.shape[1]):  # All of the possible locations
            # the indices for times which have alarmed at that location
            same = (loc == nonzero_locations)
            if(np.any(same)):  # check if this alarmed at any point
                # These are all of the times which alarmed
                alarmed_times = nonzero_times[same]
                # Determine the first alarming time
                time_to_alarm.append(min(alarmed_times))
            else:
                # this represents a location which was never alarmed
                time_to_alarm.append(BIG_NUMBER)

        time_to_alarm = np.array(time_to_alarm)
        if flip_x:
            X = max(self.X) - self.X + min(self.X)
        else:
            X = self.X  # Don't want to mutate the original

        if flip_y:
            Y = max(self.Y) - self.Y + min(self.Y)
        else:
            Y = self.Y

        if infeasible_locations is not None:
            for infeasible in infeasible_locations:
                x1, y1, x2, y2 = infeasible
                infeasible_locations = np.logical_and(np.logical_and(X > x1, X < x2), np.logical_and(Y > y1, Y < y2))
                which_infeasible = np.nonzero(infeasible_locations)
                time_to_alarm[which_infeasible] = INFEASIBLE_VALUE

        if visualize:
            cb = self.pmesh_plot(X, Y, time_to_alarm, plt, num_samples=70)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.title("Time to alarm versus location on the wall")
        return (X, Y, time_to_alarm)

    def example_time_to_alarm(self, x_bounds, y_bounds,
                              center, show=False, scale=1, offset=0):
        """
        Xs and Ys are the upper and lower bounds
        center is the x y coords
        scale is a multiplicative factor
        offset is additive
        """
        Xs = np.linspace(*x_bounds)
        Ys = np.linspace(*y_bounds)
        x, y = np.meshgrid(Xs, Ys)
        z = (x - center[0]) ** 2 + (y - center[1]) ** 2
        z = z * scale + offset
        if show:
            plt.cla()
            plt.clf()
            cb = plt.pcolormesh(x, y, z, cmap=plt.cm.inferno)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.pause(4.0)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        return (x, y, z)

    def make_lookup(self, X, Y, time_to_alarm):
        """Returns a function which searches
        the data at the sample nearest a given point
        """
        best = np.argmin(
            time_to_alarm)  # get the location of the shortest time to alarm
        XY = np.vstack((X, Y)).transpose()  # combine the x and y data points
        self.logger.info("The best time, determined by exaustive search, is {} and occurs at {}".format(
            time_to_alarm[best], XY[best, :]))
        EPSILON = 0.00000001

        def ret_func(xy):  # this is what will be returned
            diff = xy - XY  # get the x and y distances from the query point for each marked point
            dist = np.linalg.norm(diff, axis=1)
            locs = np.argsort(dist)[:4]
            # weight by one over the distance to the sample point
            weights = 1.0 / (dist[locs] + EPSILON)
            reweighted = weights * time_to_alarm[locs]
            closest_time = sum(reweighted) / sum(weights)
            #f = interpolate.interp2d(X, Y, time_to_alarm)
            return closest_time
        return ret_func

    def make_total_lookup_function(self, xytimes):
        """
        This takes as input a list of tuples in the form [(x, y, times), (x, y, times), ....] coresponding to the different smokes sources
        it returns a function mapping [x1, y1, x2, y2, ....], represnting the x, y coordinates of each detector,to the objective function value
        """
        funcs = []
        for x, y, times in xytimes:
            funcs.append(self.make_lookup(x, y, times))

        def ret_func(xys, verbose=False):
            """
            xys represents the x, y location of each of the smoke detectors
            """
            all_times = []  # each internal list coresponds to a smoke detector location
            for i in range(0, len(xys), 2):
                all_times.append([])
                x = xys[i]
                y = xys[i + 1]
                for func in funcs:
                    all_times[-1].append(func([x, y]))
            #                     np.amax([np.amin([f(x) for x in xs]) for f in functions])
            # Take the min over all locations then max over all sources
            all_times = np.asarray(all_times)
            time_for_each_source = np.amin(all_times, axis=0)
            worst_source = np.amax(time_for_each_source)
            if verbose:
                print("all of the times are {}".format(all_times))
                print("The quickest detction for each source is {}".format(
                    time_for_each_source))
                print(
                    "The slowest-to-be-detected source takes {}".format(worst_source))
            return worst_source
        return ret_func

    def plot_inputs(self, inputs, optimized):
        plt.cla()
        plt.clf()
        f, ax = self.get_square_axis(len(inputs))
        max_z = 0
        for i, (x, y, z) in enumerate(inputs):
            max_z = max(max_z, max(z)) # record this for later plotting
            cb = self.pmesh_plot(x, y, z, ax[i])
            for j in range(0, len(optimized), 2):
                detectors = ax[i].scatter(optimized[j], optimized[j + 1],
                              c='w', edgecolors='k')
                ax[i].legend([detectors], ["optimized detectors"])
        f.colorbar(cb)
        f.suptitle("The time to alarm for each of the smoke sources")
        plt.show()
        return max_z

    def get_square_axis(self, num):
        """
        arange subplots in a rough square based on the number of inputs
        """
        if num == 1:
            f, ax = plt.subplots(1,1)
            ax = np.asarray([ax])
            return f, ax
        num_x = np.ceil(np.sqrt(num))
        num_y = np.ceil(num / num_x)
        f, ax = plt.subplots(int(num_y), int(num_x))
        ax = ax.flatten() #
        return f, ax


    def plot_sweep(self, xytimes, fixed_detectors, bounds, max_val=None, centers=None):
        """
        xytimes : ArrayLike[Tuple[]]
            the smoke propagation information
        xs : ArrayLike
            [x1, y1, x2, y2...] representing the fixed location of the smoke detectors
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high] the bounds on the swept variable
        """
        # TODO refactor so this is the same as the other one
        time_func = self.make_total_lookup_function(xytimes)
        print(time_func)
        x_low, x_high, y_low, y_high = bounds
        xs = np.linspace(x_low, x_high)
        ys = np.linspace(y_low, y_high)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        grid_xs = grid_xs.flatten()
        grid_ys = grid_ys.flatten()
        grid = np.vstack((grid_xs, grid_ys)).transpose()
        print(grid.shape)
        times = []
        for xy in grid:
            locations = np.hstack((fixed_detectors, xy))
            times.append(time_func(locations))
        plt.cla()
        plt.clf()

        cb = self.pmesh_plot(grid_xs, grid_ys, times, plt, max_val)
        # even and odd points
        fixed = plt.scatter(fixed_detectors[::2], fixed_detectors[1::2], c='k')
        plt.colorbar(cb)  # Add a colorbar to a plot
        if centers is not None:
            centers = plt.scatter(centers[::2], centers[1::2], c='w')
            plt.legend([fixed, centers], [
                       "The fixed detectors", "Centers of smoke sources"])
        else:
            plt.legend([fixed], ["The fixed detectors"])
        plt.title("Effects of placing the last detector with {} fixed".format(
            int(len(fixed_detectors)/2)))
        plt.show()

    def pmesh_plot(self, xs, ys, values, plotter, max_val=None, num_samples=50, cmap=plt.cm.inferno):
        points = np.stack((xs, ys), axis=1)
        sample_points = (np.linspace(min(xs), max(xs), num_samples), np.linspace(min(ys), max(ys), num_samples))
        xis, yis = np.meshgrid(*sample_points)
        flattened_xis = xis.flatten()
        flattened_yis = yis.flatten()
        interpolated = griddata(points, values, (flattened_xis, flattened_yis))
        reshaped_interpolated = np.reshape(interpolated, xis.shape)
        if max_val is not None:
            norm = mpl.colors.Normalize(0, max_val)
        else:
            norm = mpl.colors.Normalize() # default

        cb = plotter.pcolormesh(xis, yis, reshaped_interpolated, cmap=cmap, norm=norm)
        return cb # return the colorbar

    def visualize_all(self, objective_func, optimized_detectors, bounds, max_val=None, num_samples=30, verbose=False):
        """
        The goal is to do a sweep with each of the detectors leaving the others fixed
        """
        # set up the sampling locations
        x_low, x_high, y_low, y_high = bounds
        xs = np.linspace(x_low, x_high, num_samples)
        ys = np.linspace(y_low, y_high, num_samples)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        grid_xs = grid_xs.flatten()
        grid_ys = grid_ys.flatten()
        # This is a (n, 2) where each row is a point
        grid = np.vstack((grid_xs, grid_ys)).transpose()

        # create the subplots
        plt.cla()
        plt.clf()
        #f, ax = plt.subplots(int(len(optimized_detectors)/2), 1)
        f, ax = self.get_square_axis(len(optimized_detectors)/2)

        num_samples = grid.shape[0]

        for i in range(0, len(optimized_detectors), 2):
            selected_detectors = np.concatenate(
                (optimized_detectors[:i], optimized_detectors[(i+2):]), axis=0)  # get all but one

            repeated_selected = np.tile(np.expand_dims(
                selected_detectors, axis=0), reps=(num_samples, 1))
            locations = np.concatenate((grid, repeated_selected), axis=1)

            times = [objective_func(xys) for xys in locations]
            if type(ax) is np.ndarray: # ax may be al
                which_plot = ax[int(i/2)]
            else:
                which_plot = ax

            cb = self.pmesh_plot(grid_xs, grid_ys, times, which_plot, max_val)

            fixed = which_plot.scatter(
                selected_detectors[::2], selected_detectors[1::2], c='w', edgecolors='k')

            if verbose:
                which_plot.legend([fixed], ["the fixed detectors"])
                which_plot.set_xlabel("x location")
                which_plot.set_ylabel("y location")

        f.suptitle("The effects of sweeping one detector with all other fixed")
        plt.colorbar(cb, ax=ax[-1])
        plt.show()

    def optimize(self, sources, bounds, initialization, genetic=True, visualize=True):
        """
        sources : ArrayLike
            list of (x, y, time) tuples
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high]
        initialization : ArrayLike
            [x1, y1, x2, y2,...] The initial location for the optimization
        """
        expanded_bounds = []
        for i in range(0, len(initialization), 2):
            expanded_bounds.extend(
                [(bounds[0], bounds[1]), (bounds[2], bounds[3])])
        print("The bounds are now {}".format(expanded_bounds))
        total_ret_func = self.make_total_lookup_function(sources)
        values = []
        # TODO see if there's a more efficient way to do this
        def callback(xk, convergence): # make the callback to record the values of the function
            val = total_ret_func(xk) # the objective function
            values.append(val)

        if genetic:
            res = differential_evolution(total_ret_func, expanded_bounds, callback=callback)
        else:
            res = minimize(total_ret_func, initialization, method='COBYLA', callback=callback)

        if visualize:
            plt.title("Objective function values over time")
            plt.xlabel("Number of function evaluations")
            plt.ylabel("Objective function")
            plt.plot(values)
            plt.show()
            max_val = self.plot_inputs(sources, res.x)
            self.visualize_all(total_ret_func, res.x, bounds, max_val=max_val)
        xs = res.x
        output = "The locations are: "
        for i in range(0, xs.shape[0], 2):
            output += ("({:.3f}, {:.3f}), ".format(xs[i], xs[i+1]))
        print(output)
        return res


if __name__ == "__main__":  # Only run if this was run from the commmand line
    SDO = SDOptimizer()
    SDO.load_data(DATA_FILE)  # Load the data file
    X1, Y1, time_to_alarm1 = SDO.get_time_to_alarm(False)
    X2, Y2, time_to_alarm2 = SDO.example_time_to_alarm(
        (0, 1), (0, 1), (0.3, 0.7), False)
    ret_func = SDO.make_lookup(X1, Y1, time_to_alarm1)
    total_ret_func = SDO.make_total_lookup_function(
        [(X1, Y1, time_to_alarm1), (X2, Y2, time_to_alarm2)])

    CENTERS = [0.2, 0.8, 0.8, 0.8, 0.8, 0.2]

    x1, y1, z1 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[0:2], False)
    x2, y2, z2 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[2:4], False)
    x3, y3, z3 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[4:6], False)
    inputs = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    total_ret_func = SDO.make_total_lookup_function(inputs)
    BOUNDS = ((0, 1), (0, 1), (0, 1), (0, 1))  # constraints on inputs
    INIT = (0.51, 0.52, 0.47, 0.6, 0.55, 0.67)
    res = minimize(total_ret_func, INIT, method='COBYLA')
    print(res)
    x = res.x
