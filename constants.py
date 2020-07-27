"""Configuration parameters likely to be shared across scripts."""
from os.path import abspath, join, dirname

# This represents the current
this_dir_path = dirname(abspath(__file__))
# This is the directory the data is in.
data_dir = join(this_dir_path, "data")

# The folder names for each of the data folders
SMOKE_FOLDERS = ["first_computer_full3D",
                 "second_computer_full3D", "third_computer_full3D"]

# prepend the absolute path of the data directory
SMOKE_FOLDERS = [join(data_dir, smoke_folder)
                 for smoke_folder in SMOKE_FOLDERS]

# The folder names for each of the false alarm folders
FALSE_ALARM_FOLDERS = ["bike_full3D"]

# prepend the absolute path of the data directory
FALSE_ALARM_FOLDERS = [join(data_dir, false_alarm_folder)
                       for false_alarm_folder in FALSE_ALARM_FOLDERS]
