#!/usr/bin/env python
# coding: utf-8
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import scipy
import pdb
import logging
import glob
import os
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
from scipy.interpolate import griddata
from mpl_toolkits import mplot3d
#from platypus import NSGAII, Problem, Real, Binary, Integer
from platypus import * # TODO fix this terrible practice
from tqdm.notebook import trange, tqdm # For plotting progress
from time import sleep


DATA_FILE = "exportUSLab.csv"  # Points to the data Katie gave us
VISUALIZE = False
ALARM_THRESHOLD = 4e-20
BIG_NUMBER = 100
INFEASIBLE_VALUE = BIG_NUMBER * 1.5


class SDOptimizer():
    def __init__(self):
        self.logger = logging.getLogger('main')
        self.logger.debug("Instantiated the optimizer")
        self.is3d = False


    def load_data(self, data_file):
        """
        data_file : string
            This is the path to the data file as exported by the Fluent simulation
        """
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

    def load_directory(self, directory):
        """
        directory : String
            The folder containing the data files
        -----returns-----
        data : List[Tuple[ArrayLike]]
            This is the the x, y, time to alarm data for each data file
        """
        files_pattern = os.path.join(directory, "*")
        # this should match all files in the directory
        files = glob.glob(files_pattern)
        data = []
        for file in files:
            data.append(self.load_data(file))
        return data

    def visualize(self, show=False):
        """
        TODO update this so it outputs a video
        """
        print(self.max_consentrations)
        max_consentration = max(self.max_consentrations)
        print(max_consentration)

        ims = []

        for i, df in enumerate(self.all_times):
            plt.cla()
            plt.clf()
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.title("consentration at timestep {} versus position".format(i))
            norm = mpl.colors.Normalize(vmin=0, vmax=1.0)

            cb = self.pmesh_plot(
                df['X'],
                df['Y'],
                df['C'],
                plt,
                max_val=max_consentration)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.savefig("vis/consentration{:03d}.png".format(i))
            if show:
                plt.show()

    def get_time_to_alarm(
            self,
            flip_x=False,
            flip_y=False,
            infeasible_locations=None,
            alarm_threshold=ALARM_THRESHOLD,
            visualize=False):
        """
        file_x, flip_y : Boolean
            Should the data be flipped about the corresponding axis for augmentation
        infeasible_locations : ArrayLike[Tuple[Float]]
            An array of tuples where each one represents an object [(x1, y1, x2, y2),....]
        alarm_threshold : Float
            What consentraion will trigger the detector
        visualize : Boolean
            Should it be shown
        """
        consentrations = np.asarray(
            [x['C'].values for x in self.all_times])  # Get all of the consentrations
        self.logger.info(
            "There are {} timesteps and {} flattened locations".format(
                consentrations.shape[0],
                consentrations.shape[1]))

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
                infeasible_locations = np.logical_and(np.logical_and(
                    X > x1, X < x2), np.logical_and(Y > y1, Y < y2))
                which_infeasible = np.nonzero(infeasible_locations)
                time_to_alarm[which_infeasible] = INFEASIBLE_VALUE

        if visualize:
            cb = self.pmesh_plot(X, Y, time_to_alarm, plt, num_samples=70)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.title("Time to alarm versus location on the wall")
            plt.xlabel("X location")
            plt.ylabel("Y location")
            plt.show()
            samples = np.random.choice(consentrations.shape[1], 10)
            rows = consentrations[:,samples].transpose()
            for row in rows:
                plt.plot(row)
            plt.title("random samples of consentration over time")
            plt.xlabel("timesteps")
            plt.ylabel("consentration")
            plt.show()

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
        """
        X, Y : ArrayLike[Float]
            The x, y locations of the data points from the simulation
        time_to_alarm : ArrayLike[Float]
            The time to alarm corresponding to each of the locations
        -----returns-----
        the data at the sample nearest a given point
        """
        best = np.argmin(
            time_to_alarm)  # get the location of the shortest time to alarm
        XY = np.vstack((X, Y)).transpose()  # combine the x and y data points
        self.logger.info("The best time, determined by exaustive search, is {} and occurs at {}".format(
            time_to_alarm[best], XY[best, :]))
        EPSILON = 0.00000001

        def ret_func(xy):  # this is what will be returned
            if False:#TODO this should be True
                closest_time = griddata(XY, time_to_alarm, xy)
                print(closest_time)
            else:
                diff = xy - XY  # get the x and y distances from the query point for each marked point
                dist = np.linalg.norm(diff, axis=1)
                locs = np.argsort(dist)[:1]
                # weight by one over the distance to the sample point
                weights = 1.0 / (dist[locs] + EPSILON)
                reweighted = weights * time_to_alarm[locs]
                closest_time = sum(reweighted) / sum(weights)
            return closest_time
        return ret_func

    def make_total_lookup_function(
            self,
            xytimes,
            verbose=False,
            type="worst_case",
            masked=False):
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
        -----returns-----
        ret_func : Function[ArrayLike[Float] -> Float]
            This is the function which will eventually be optimized and it maps from the smoke detector locations to the time to alarm
            A function mapping [x1, y1, x2, y2, ....], represnting the x, y coordinates of each detector,to the objective function value
        """
        # Create data which will be used inside of the function to be returned
        funcs = []
        for x, y, times in xytimes:
            # create all of the functions mapping from a location to a time
            funcs.append(self.make_lookup(x, y, times))

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
                    if on[0]: # don't evaluate a detector which isn't on, on is really a list of length 1
                        all_times.append([])
                        some_on = True
                        for func in funcs:
                            all_times[-1].append(func([x, y]))
                all_times = np.asarray(all_times)
                if not some_on: # This means that no sources were turned on
                    return BIG_NUMBER
            else:
                for i in range(0, len(xys), 2):
                    all_times.append([])
                    x, y= xys[i:i+2]
                    for func in funcs:
                        all_times[-1].append(func([x, y]))
                all_times = np.asarray(all_times)

            if type == "worst_case":
                time_for_each_source = np.amin(all_times, axis=0)
                worst_source = np.amax(time_for_each_source)
                ret_val = worst_source
            if type == "second":
                time_for_each_source = np.amin(all_times, axis=0)
                second_source = np.sort(time_for_each_source)[1]
                ret_val = second_source
            if type == "softened":
                time_for_each_source = np.amin(all_times, axis=0)
                sorted = np.sort(time_for_each_source)[1]
                ALPHA = 0.3
                ret_val = (sorted[0] + ALPHA * sorted[1]) / (1 + ALPHA)

            if verbose:
                print("all of the times are {}".format(all_times))
                print("The quickest detction for each source is {}".format(
                    time_for_each_source))
                print(
                    "The slowest-to-be-detected source takes {}".format(worst_source))
            return ret_val
        return ret_func

    def make_platypus_objective_function(self, sources):
        total_ret_func = self.make_total_lookup_function(sources) # the function to be optimized
        def multiobjective_func(x): # this is the double objective function
            #return [total_ret_func(x), np.linalg.norm(x)]
            return [total_ret_func(x), np.linalg.norm(x)]

        num_inputs = len(sources) * 2 # there is an x, y for each source
        NUM_OUPUTS = 2 # the default for now
        problem = Problem(num_inputs, NUM_OUPUTS) # define the demensionality of input and output spaces
        x, y, time = sources[0] # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print("min x : {}, max x : {}, min y : {}, max y : {}".format(min_x, max_x, min_y, max_y))
        problem.types[::2] = Real(min_x, max_x) # This is the feasible region
        problem.types[1::2] = Real(min_y, max_y)
        problem.function = multiobjective_func
        return problem

    def make_platypus_mixed_integer_objective_function(self, sources):
        total_ret_func = self.make_total_lookup_function(sources, masked=True) # the function to be optimized
        location_func  = self.make_location_objective(masked=True)
        def multiobjective_func(x): # this is the double objective function
            return [total_ret_func(x), location_func(x)]

        num_inputs = len(sources) * 3 # there is an x, y, and a mask for each source
        NUM_OUPUTS = 2 # the default for now
        problem = Problem(num_inputs, NUM_OUPUTS) # define the demensionality of input and output spaces
        x, y, time = sources[0] # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print("min x : {}, max x : {}, min y : {}, max y : {}".format(min_x, max_x, min_y, max_y))
        problem.types[0::3] = Real(min_x, max_x) # This is the feasible region
        problem.types[1::3] = Real(min_y, max_y)
        problem.types[2::3] = Binary(1) # This appears to be inclusive, so this is really just (0, 1)
        problem.function = multiobjective_func
        return problem

    def make_location_objective(self, masked):
        """
        an example function to evalute the quality of the locations
        """
        if masked:
            def location_evaluation(xyons): # TODO make this cleaner
                good = []
                for i in range(0, len(xyons), 3):
                    x, y, on = xyons[i:i+3]
                    if on[0]:
                        good.extend([x,y])
                if len(good) == 0: # This means none were on
                    return 0
                else:
                    return np.linalg.norm(good)
        else:
            def location_evaluation(xys):
                return np.linalg.norm(xys)


        return location_evaluation

    def plot_inputs(self, inputs, optimized):
        plt.cla()
        plt.clf()
        f, ax = self.get_square_axis(len(inputs))
        max_z = 0
        for i, (x, y, z) in enumerate(inputs):
            max_z = max(max_z, max(z))  # record this for later plotting
            cb = self.pmesh_plot(x, y, z, ax[i])
            for j in range(0, len(optimized), 2):
                detectors = ax[i].scatter(optimized[j], optimized[j + 1],
                                          c='w', edgecolors='k')
                ax[i].legend([detectors], ["optimized detectors"])
        f.colorbar(cb)
        f.suptitle("The time to alarm for each of the smoke sources")
        plt.show()
        return max_z

    def get_square_axis(self, num, is_3d=False):
        """
        arange subplots in a rough square based on the number of inputs
        """
        if num == 1:
            if is_3d:
                f, ax = plt.subplots(1, 1, projection='3d')
            else:
                f, ax = plt.subplots(1, 1)

            ax = np.asarray([ax])
            return f, ax
        num_x = np.ceil(np.sqrt(num))
        num_y = np.ceil(num / num_x)
        if is_3d:
            f, ax = plt.subplots(int(num_y), int(num_x),  projection='3d')
        else:
            f, ax = plt.subplots(int(num_y), int(num_x))

        ax = ax.flatten()
        return f, ax

    def plot_sweep(self, xytimes, fixed_detectors,
                   bounds, max_val=None, centers=None):
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
            int(len(fixed_detectors) / 2)))
        plt.show()

    def pmesh_plot(
            self,
            xs,
            ys,
            values,
            plotter,
            max_val=None,
            num_samples=50,
            is_3d=False,
            cmap=plt.cm.inferno):
        """
        conveneince function to easily plot the sort of data we have
        """
        points = np.stack((xs, ys), axis=1)
        sample_points = (np.linspace(min(xs), max(xs), num_samples),
                         np.linspace(min(ys), max(ys), num_samples))
        xis, yis = np.meshgrid(*sample_points)
        flattened_xis = xis.flatten()
        flattened_yis = yis.flatten()
        interpolated = griddata(points, values, (flattened_xis, flattened_yis))
        reshaped_interpolated = np.reshape(interpolated, xis.shape)
        if max_val is not None:
            norm = mpl.colors.Normalize(0, max_val)
        else:
            norm = mpl.colors.Normalize()  # default

        if self.is3d:
            plt.cla()
            plt.clf()
            plt.close()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #cb = ax.plot_surface(xis, yis, reshaped_interpolated,cmap=cmap, norm=norm, edgecolor='none')
            cb = ax.contour3D(xis, yis, reshaped_interpolated, 60, cmap=cmap)
            plt.show()
        else:
            cb = plotter.pcolormesh(
                xis, yis, reshaped_interpolated, cmap=cmap, norm=norm)
        return cb  # return the colorbar

    def plot_3d(
            self,
            xs,
            ys,
            values,
            plotter,
            max_val=None,
            num_samples=50,
            is_3d=False,
            cmap=plt.cm.inferno):
        """
        conveneince function to easily plot the sort of data we have
        """
        points = np.stack((xs, ys), axis=1)
        sample_points = (np.linspace(min(xs), max(xs), num_samples),
                         np.linspace(min(ys), max(ys), num_samples))
        xis, yis = np.meshgrid(*sample_points)
        flattened_xis = xis.flatten()
        flattened_yis = yis.flatten()
        interpolated = griddata(points, values, (flattened_xis, flattened_yis))
        reshaped_interpolated = np.reshape(interpolated, xis.shape)
        if max_val is not None:
            norm = mpl.colors.Normalize(0, max_val)
        else:
            norm = mpl.colors.Normalize()  # default

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cb = ax.plot_surface(xis, yis, reshaped_interpolated,cmap=cmap, norm=norm, edgecolor='none')
        ax.set_title('Surface plot')
        plt.show()
        return cb  # return the colorbar

    def visualize_all(self, objective_func, optimized_detectors,
                      bounds, max_val=None, num_samples=30, verbose=False, is3d=False):
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
        f, ax = self.get_square_axis(len(optimized_detectors) / 2)

        num_samples = grid.shape[0]

        for i in range(0, len(optimized_detectors), 2):
            selected_detectors = np.concatenate(
                (optimized_detectors[:i], optimized_detectors[(i + 2):]), axis=0)  # get all but one

            repeated_selected = np.tile(np.expand_dims(
                selected_detectors, axis=0), reps=(num_samples, 1))
            locations = np.concatenate((grid, repeated_selected), axis=1)

            times = [objective_func(xys) for xys in locations]
            if isinstance(ax, np.ndarray):  # ax may be al
                which_plot = ax[int(i / 2)]
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

    def optimize(self, sources, bounds, initialization,
                 genetic=True, platypus=False, visualize=True, is3d=False, masked=False, **kwargs):
        """
        sources : ArrayLike
            list of (x, y, time) tuples
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high]
        initialization : ArrayLike
            [x1, y1, x2, y2,...] The initial location for the optimization
        genetic : Boolean
            whether to use a genetic algorithm
        masked : Boolean
            Whether the input is masked
        kwargs : This is some python dark majic stuff which effectively lets you get a dictionary of named arguments
        """
        expanded_bounds = []
        for i in range(0, len(initialization), 2):
            expanded_bounds.extend(
                [(bounds[0], bounds[1]), (bounds[2], bounds[3])]) # set up the appropriate number of bounds
        if "type" in kwargs:
            total_ret_func = self.make_total_lookup_function(sources, type=kwargs["type"]) # the function to be optimized
        else:
            total_ret_func = self.make_total_lookup_function(sources) # the function to be optimized
        if platypus:
            if masked:
                problem = self.make_platypus_mixed_integer_objective_function(sources) #TODO remove this
                # it complains about needing a defined mutator for mixed problems
                # Suggestion taken from https://github.com/Project-Platypus/Platypus/issues/31
                algorithm = NSGAII(problem, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
            else:
                problem = self.make_platypus_objective_function(sources) #TODO remove this
                algorithm = NSGAII(problem)
            # optimize the problem using 1,000 function evaluations
            algorithm.run(1000)
            for solution in algorithm.result:
                print("Solution : {}, Location : {}".format(solution.objectives, solution.variables))

            plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
            plt.xlabel("The real function")
            plt.ylabel("The norm of the values")
            plt.title("Pareto optimality curve for the two functions")
            plt.show()

        values = []
        # TODO see if there's a more efficient way to do this

        def callback(xk, convergence):  # make the callback to record the values of the function
            val = total_ret_func(xk)  # the objective function
            values.append(val)

        if genetic:
            res = differential_evolution( # this is a genetic algorithm implementation
                total_ret_func, expanded_bounds, callback=callback)
        else:
            res = minimize(total_ret_func, initialization,
                           method='COBYLA', callback=callback)

        if visualize:
            plt.title("Objective function values over time")
            plt.xlabel("Number of function evaluations")
            plt.ylabel("Objective function")
            plt.plot(values)
            plt.show()
            max_val = self.plot_inputs(sources, res.x)
            self.visualize_all(total_ret_func, res.x, bounds, max_val=max_val)
            xs = res.x
            print("The bounds are now {}".format(expanded_bounds))
            output = "The locations are: "
            for i in range(0, xs.shape[0], 2):
                output += ("({:.3f}, {:.3f}), ".format(xs[i], xs[i + 1]))
            print(output)
        return res

    def evaluate_optimization(self, sources, bounds, initialization,
                 genetic=True, visualize=True, num_iterations=10):
        vals = []
        locs = []
        iterations = []
        for i in trange(num_iterations):
            res = self.optimize(sources, bounds, initialization, genetic=genetic, visualize=False)
            vals.append(res.fun)
            locs.append(res.x)
            iterations.append(res.nit)

        if visualize:
            self.show_optimization_statistics(vals, iterations, locs)

        return vals, locs, iterations

    def show_optimization_statistics(self, vals, iterations, locs):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        #ax1.hist(vals, bins=10, rwidth=0.25) # plot the bins as a quarter of the spread
        ax1.violinplot(vals)
        ax1.set_ylabel("objective function values") # TODO make this a histogram
        #ax2.hist(iterations, bins=10)
        ax2.violinplot(iterations)
        ax2.set_ylabel("number of iterations function values") # TODO make this a histogram
        plt.show()
        for loc in locs:
            self.plot_xy(loc)
        plt.xlim(0, 8.1)
        plt.ylim(0, 3)
        plt.show()


    def plot_xy(self, xy):
        plt.scatter(xy[::2], xy[1::2])

    def set_3d(self, value=False):
        """
        set whether it should be 3d
        """
        self.is3d = value

    def test_tqdm(self):
        for _ in trange(30): # For plotting progress
            sleep(0.5)



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
