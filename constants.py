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

SMOKE_FOLDERS_LONG = ["laptop_1000_steps",
                      "wall_computer_1_1000_steps",
                      "wall_computer_2_1000_steps"]

# prepend the absolute path of the data directory
SMOKE_FOLDERS_LONG = [join(data_dir, "long_runs", smoke_folder)
                      for smoke_folder in SMOKE_FOLDERS_LONG]


# The folder names for each of the false alarm folders
FALSE_ALARM_FOLDERS = ["bike_full3D"]

# prepend the absolute path of the data directory
FALSE_ALARM_FOLDERS = [join(data_dir, "long_runs", false_alarm_folder)
                       for false_alarm_folder in FALSE_ALARM_FOLDERS]

FALSE_ALARM_FOLDERS_LONG = ["bike_1000_steps"]

# prepend the absolute path of the data directory
FALSE_ALARM_FOLDERS_LONG = [join(data_dir, false_alarm_folder)
                            for false_alarm_folder in FALSE_ALARM_FOLDERS_LONG]

NASA_DETECTORS = ((2.11500001, -3., -2.84649992), (6.42500019, 0., -2.84649992))
SOURCE_LOCATIONS=((1.9866, -0.5119, -1.1846), (2.2185, -2.5768, -1.0527), (7.279, -3.0, -1.1963))

MESH_FILE = join(data_dir, "surface_mesh.stl")
