import os
DATA_FILE = "exportUSLab.csv"  # Points to the data Katie gave us

# Make this cross platform by using the right seperating character for the
# current operating system
SMOKE_FOLDERS = [os.path.join("data", "first_computer_full3D"),
                 os.path.join("data", "second_computer_full3D"),
                 os.path.join("data", "third_computer_full3D")]

FALSE_ALARM_FOLDERS = [os.path.join("data", "bike_full3D")]

VISUALIZE = False
#ALARM_THRESHOLD = 4e-20
# This is the NASA-stated
ALARM_THRESHOLD = 0.5e-6
BIG_NUMBER = 1000
# The number of timesteps times this is what will be recored if it never alarms
NEVER_ALARMED_MULTIPLE = 1.5
# The number of timesteps times this is what will be recored if the region is infeasible
INFEASIBLE_MULTIPLE = 3
PAPER_READY = True
PLOT_TITLES = False
EPSILON = 0.000000001

# Should plots by default be interpolated?
SMOOTH_PLOTS = True

# Change how sample points are interpolated
INTERPOLATION_METHOD = "nearest"  # "linear" "cubic"
