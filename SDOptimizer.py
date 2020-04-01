#!/usr/bin/env python
# coding: utf-8
from constants import *  # TODO fix this
from functions import make_location_objective, make_counting_objective, make_lookup, make_total_lookup_function
from visualization import show_optimization_statistics
from time import sleep
from tqdm.notebook import trange, tqdm  # For plotting progress
from platypus import *  # TODO fix this terrible practice
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
import os
import glob
import logging
import pdb
import scipy
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import matplotlib
import warnings
import copy
matplotlib.use('module://ipykernel.pylab.backend_inline')
#from platypus import NSGAII, Problem, Real, Binary, Integer


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

    def load_timestep_directory(self, directory):
        """
        directory : String
            The folder containing the data files
            Each of the files should represent a timestep and they must be
            alphabetized
        -----returns-----
        data : TODO
        """
        files_pattern = os.path.join(directory, "*")
        filenames = sorted(glob.glob(files_pattern))
        if len(filenames) == 0:
            raise ValueError(
                "There were no files in the specified directory : {}".format(directory))
        self.all_times = []
        self.max_consentrations = []
        for filename in filenames:
            df = pd.read_csv(filename, sep=' ', skipinitialspace=True)
            df.drop(labels=df.columns[-1], inplace=True, axis=1)
            df = df.rename(
                columns={
                    'nodenumber': 'N',
                    'x-coordinate': 'X',
                    'y-coordinate': 'Y',
                    'z-coordinate': 'Z',
                    'dpm-concentration': 'C'})
            consentration = df['C'].values
            self.max_consentrations.append(np.max(consentration))
            self.all_times.append(df)
        self.X = df['X'].values
        # yes, this is Z. Must be a different reference frame
        self.Y = df['Z'].values

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

    def visualize(self, show=False, log=True):
        """
        TODO update this so it outputs a video
        show : Boolean
            Do a matplotlib plot of every frame
        log : Boolean
            plot the consentration on a log scale
        """
        max_consentration = max(self.max_consentrations)

        print("Writing output files to ./vis")
        for i, df in tqdm(
            enumerate(
                self.all_times), total=len(
                self.all_times)):  # this is just wrapping it in a progress bar
            plt.cla()
            plt.clf()
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.title("consentration at timestep {} versus position".format(i))
            norm = mpl.colors.Normalize(vmin=0, vmax=1.0)

            cb = self.pmesh_plot(
                self.X,
                self.Y,
                df['C'],
                plt,
                log=log,
                max_val=max_consentration)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.savefig("vis/consentration{:03d}.png".format(i))
            if show:
                plt.show()

    def get_time_to_alarm(
            self,
            flip_x=False,
            flip_y=False,
            x_shift=0,
            y_shift=0,
            infeasible_locations=None,
            alarm_threshold=ALARM_THRESHOLD,
            visualize=False,
            num_samples_visualized=10):
        """
        file_x, flip_y : Boolean
            Should the data be flipped about the corresponding axis for augmentation
        infeasible_locations : ArrayLike[Tuple[Float]]
            An array of tuples where each one represents an object [(x1, y1, x2, y2),....]
        alarm_threshold : Float
            What consentraion will trigger the detector
        visualize : Boolean
            Should it be shown
        num_samples_visualized : into
            the number of scattered points to visualize the concentration over time for
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
            X = copy.copy(self.X)  # Don't want to mutate the original

        if flip_y:
            Y = max(self.Y) - self.Y + min(self.Y)
        else:
            Y = copy.copy(self.Y)

        X += x_shift
        Y += y_shift
        if infeasible_locations is not None:
            for infeasible in infeasible_locations:
                x1, y1, x2, y2 = infeasible
                infeasible_locations = np.logical_and(np.logical_and(
                    X > x1, X < x2), np.logical_and(Y > y1, Y < y2))
                which_infeasible = np.nonzero(infeasible_locations)
                time_to_alarm[which_infeasible] = INFEASIBLE_VALUE


        if visualize:
            cb = self.pmesh_plot(
                X,
                Y,
                time_to_alarm,
                plt,
                num_samples=70,
                cmap=mpl.cm.Greys)  # choose grey to plot color over
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.title("Time to alarm versus location on the wall")
            plt.xlabel("X location")
            plt.ylabel("Y location")
            samples = np.random.choice(
                consentrations.shape[1],
                num_samples_visualized)
            # plot the sampled locations
            xs = X[samples]
            ys = Y[samples]
            for x_, y_ in zip(xs, ys):  # dashed to avoid confusion
                plt.scatter(x_, y_)
            plt.show()
            rows = consentrations[:, samples].transpose()
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

    def make_platypus_objective_function(
            self, sources, type="basic", bad_sources=[]):
        """
        sources

        bad_sources : ArrayLike[Sources]
            These are the ones which we don't want to be near
        """
        if type == "basic":
            raise NotImplementedError("I'm not sure I'll ever do this")
            # return self.make_platypus_objective_function_basic(sources)
        elif type == "counting":
            return self.make_platypus_objective_function_counting(sources)
        elif type == "competing_function":
            return self.make_platypus_objective_function_competing_function(
                sources, bad_sources)
        else:
            raise ValueError("The type : {} is not an option".format(type))

    def make_platypus_objective_function_competing_function(
            self, sources, bad_sources=[]):
        total_ret_func = make_total_lookup_function(
            sources)  # the function to be optimized
        bad_sources_func = make_total_lookup_function(
            bad_sources, type="fastest"
        )  # the function to be optimized

        def multiobjective_func(x):  # this is the double objective function
            return [total_ret_func(x), bad_sources_func(x)]

        num_inputs = len(sources) * 2  # there is an x, y for each source
        NUM_OUPUTS = 2  # the default for now
        # define the demensionality of input and output spaces
        problem = Problem(num_inputs, NUM_OUPUTS)
        x, y, time = sources[0]  # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print(
            "min x : {}, max x : {}, min y : {}, max y : {}".format(
                min_x,
                max_x,
                min_y,
                max_y))
        problem.types[::2] = Real(min_x, max_x)  # This is the feasible region
        problem.types[1::2] = Real(min_y, max_y)
        problem.function = multiobjective_func
        # the second function should be maximized rather than minimized
        problem.directions[1] = Problem.MAXIMIZE
        return problem

    def make_platypus_objective_function_counting(
            self, sources, times_more_detectors=1):
        """
        This balances the number of detectors with the quality of the outcome
        """
        total_ret_func = make_total_lookup_function(
            sources, masked=True)  # the function to be optimized
        counting_func = make_counting_objective()

        def multiobjective_func(x):  # this is the double objective function
            return [total_ret_func(x), counting_func(x)]

        # there is an x, y, and a mask for each source so there must be three
        # times more input variables
        # the upper bound on the number of detectors n times the number of
        # sources
        num_inputs = len(sources) * 3 * times_more_detectors
        NUM_OUPUTS = 2  # the default for now
        # define the demensionality of input and output spaces
        problem = Problem(num_inputs, NUM_OUPUTS)
        x, y, time = sources[0]  # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print(
            "min x : {}, max x : {}, min y : {}, max y : {}".format(
                min_x,
                max_x,
                min_y,
                max_y))
        problem.types[0::3] = Real(min_x, max_x)  # This is the feasible region
        problem.types[1::3] = Real(min_y, max_y)
        # This appears to be inclusive, so this is really just (0, 1)
        problem.types[2::3] = Binary(1)
        problem.function = multiobjective_func
        return problem

    def plot_inputs(self, inputs, optimized, show_optimal=False):
        plt.cla()
        plt.clf()
        f, ax = self.get_square_axis(len(inputs))
        max_z = 0
        for i, (x, y, z) in enumerate(inputs):
            max_z = max(max_z, max(z))  # record this for later plotting
            cb = self.pmesh_plot(x, y, z, ax[i])
            if show_optimal:
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
            f, ax = plt.subplots(int(num_y), int(num_x), projection='3d')
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
        time_func = make_total_lookup_function(xytimes)
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
            log=False,  # log scale for plotting
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
            if log:
                EPSILON = 0.0000000001
                norm = mpl.colors.LogNorm(
                    EPSILON, max_val + EPSILON)  # avoid zero values
                reshaped_interpolated += EPSILON
            else:
                norm = mpl.colors.Normalize(0, max_val)
        else:
            if log:
                norm = mpl.colors.LogNorm()
                EPSILON = 0.0000000001
                reshaped_interpolated += EPSILON
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
        cb = ax.plot_surface(
            xis,
            yis,
            reshaped_interpolated,
            cmap=cmap,
            norm=norm,
            edgecolor='none')
        ax.set_title('Surface plot')
        plt.show()
        return cb  # return the colorbar

    def visualize_all(
            self,
            objective_func,
            optimized_detectors,
            bounds,
            max_val=None,
            num_samples=30,
            verbose=False,
            is3d=False,
            log=False):
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

            cb = self.pmesh_plot(
                grid_xs,
                grid_ys,
                times,
                which_plot,
                max_val,
                log=log)

            fixed = which_plot.scatter(
                selected_detectors[::2], selected_detectors[1::2], c='w', edgecolors='k')

            if verbose:
                which_plot.legend([fixed], ["the fixed detectors"])
                which_plot.set_xlabel("x location")
                which_plot.set_ylabel("y location")

        f.suptitle("The effects of sweeping one detector with all other fixed")
        plt.colorbar(cb, ax=ax[-1])
        plt.show()

    def optimize(self, sources, num_sources, bounds=None,
                 genetic=True, platypus=False, visualize=True,
                 verbose=False, is3d=False, multiobjective_type="counting",
                 intialization=None, **kwargs):
        """
        sources : ArrayLike
            list of (x, y, time) tuples
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high], will be computed from self.X, self.Y if None
        initialization : ArrayLike
            [x1, y1, x2, y2,...] The initial location for the optimization
        genetic : Boolean
            whether to use a genetic algorithm
        masked : Boolean
            Whether the input is masked TODO figure out what I meant
        platypus : Boolean
            Should really be called multiobjective. Runs multiobjective
        kwargs : This is some python dark magic stuff which effectively lets you get a dictionary of named arguments
        """

        # TODO validate that this is correct in all cases
        if bounds is None:
            X = sources[0][0]
            Y = sources[0][1]
            min_x = np.min(X)
            max_x = np.max(X)
            min_y = np.min(Y)
            max_y = np.max(Y)
            bounds = [min_x, max_x, min_y, max_y]

        expanded_bounds = []
        for i in range(0, num_sources * 2, 2):
            # set up the appropriate number of bounds
            expanded_bounds.extend(
                [(bounds[0], bounds[1]), (bounds[2], bounds[3])])
        if "type" in kwargs:
            total_ret_func = make_total_lookup_function(
                sources, type=kwargs["type"])  # the function to be optimized
        else:
            total_ret_func = make_total_lookup_function(
                sources)  # the function to be optimized
        if platypus:
            if multiobjective_type == "counting":
                problem = self.make_platypus_objective_function_counting(
                    sources, "counting")  # TODO remove this
                # it complains about needing a defined mutator for mixed problems
                # Suggestion taken from
                # https://github.com/Project-Platypus/Platypus/issues/31
                algorithm = NSGAII(
                    problem, variator=CompoundOperator(
                        SBX(), HUX(), PM(), BitFlip()))
                second_objective = "The number of detectors"
            elif multiobjective_type == "competing_function":
                if "bad_sources" not in kwargs:
                    raise ValueError(
                        "bad_sources should have been included in the kwargs")
                bad_sources = kwargs["bad_sources"]
                problem = self.make_platypus_objective_function(
                    sources, "competing_function", bad_sources=bad_sources)  # TODO remove this
                algorithm = NSGAII(problem)
                second_objective = "The time to alarm for the exercise equiptment"
            else:
                raise ValueError(
                    "The type : {} was not valid".format(multiobjective_type))
            # optimize the problem using 1,000 function evaluations
            algorithm.run(1000)
            if verbose:
                for solution in algorithm.result:
                    print(
                        "Solution : {}, Location : {}".format(
                            solution.objectives,
                            solution.variables))

            plt.scatter([s.objectives[0] for s in algorithm.result],
                        [s.objectives[1] for s in algorithm.result])
            plt.xlabel("The time to alarm")
            plt.ylabel(second_objective)
            plt.title("Pareto optimality curve for the two functions")
            plt.show()
            res = algorithm
            if visualize:
                warnings.warn(
                    "Can't visualize the objective values for a multiobjective run",
                    UserWarning)
        else:
            values = []
            # TODO see if there's a more efficient way to do this

            def callback(xk, convergence):  # make the callback to record the values of the function
                val = total_ret_func(xk)  # the objective function
                values.append(val)

            if genetic:
                res = differential_evolution(  # this is a genetic algorithm implementation
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
                self.visualize_all(
                    total_ret_func, res.x, bounds, max_val=max_val)
                xs = res.x
                print("The bounds are now {}".format(expanded_bounds))
                output = "The locations are: "
                for i in range(0, xs.shape[0], 2):
                    output += ("({:.3f}, {:.3f}), ".format(xs[i], xs[i + 1]))
                print(output)
        return res

    def evaluate_optimization(self, sources, num_sources, bounds=None,
                              genetic=True, visualize=True, num_iterations=10):
        vals = []
        locs = []
        iterations = []
        for i in trange(num_iterations):
            res = self.optimize(
                sources,
                num_sources,
                bounds=bounds,
                genetic=genetic,
                visualize=False)
            vals.append(res.fun)
            locs.append(res.x)
            iterations.append(res.nit)

        if visualize:
            show_optimization_statistics(vals, iterations, locs)

        return vals, locs, iterations

    def show_optimization_statistics(self, vals, iterations, locs):
        show_optimization_statistics(vals, iterations, locs)

    def set_3d(self, value=False):
        """
        set whether it should be 3d
        """
        self.is3d = value

    def test_tqdm(self):
        for _ in trange(30):  # For plotting progress
            sleep(0.5)


if __name__ == "__main__":  # Only run if this was run from the commmand line
    SDO = SDOptimizer()
    SDO.load_data(DATA_FILE)  # Load the data file
    X1, Y1, time_to_alarm1 = SDO.get_time_to_alarm(False)
    X2, Y2, time_to_alarm2 = SDO.example_time_to_alarm(
        (0, 1), (0, 1), (0.3, 0.7), False)
    ret_func = make_lookup(X1, Y1, time_to_alarm1)
    total_ret_func = make_total_lookup_function(
        [(X1, Y1, time_to_alarm1), (X2, Y2, time_to_alarm2)])

    CENTERS = [0.2, 0.8, 0.8, 0.8, 0.8, 0.2]

    x1, y1, z1 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[0:2], False)
    x2, y2, z2 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[2:4], False)
    x3, y3, z3 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[4:6], False)
    inputs = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    total_ret_func = make_total_lookup_function(inputs)
    BOUNDS = ((0, 1), (0, 1), (0, 1), (0, 1))  # constraints on inputs
    INIT = (0.51, 0.52, 0.47, 0.6, 0.55, 0.67)
    res = minimize(total_ret_func, INIT, method='COBYLA')
    print(res)
    x = res.x
