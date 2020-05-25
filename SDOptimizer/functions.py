import numpy as np
import pdb
from SDOptimizer.constants import *


def make_location_objective(masked):
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


def make_lookup(X, Y, time_to_alarm):
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
    #self.logger.info("The best time, determined by exaustive search, is {} and occurs at {}".format(
        #time_to_alarm[best], XY[best, :]))
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
        funcs.append(make_lookup(x, y, times))
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
        elif type=="fastest":
            #print(all_times)
            ret_val = np.amin(all_times) # this just cares about the source-detector pair that alarms fastest
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


def normalize(x, lower_bound=0, scale=1):
    if scale <= 0:
        raise ValueError("scale was less than or equal to 0")
    minimum = np.min(x)
    maximum = np.max(x)
    diff = maximum - minimum
    x_prime = (x - minimum) / diff
    x_prime = x_prime * scale + lower_bound
    return x_prime
