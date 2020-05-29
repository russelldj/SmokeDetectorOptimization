import numpy as np
import SDOptimizer as SDOmodule
from SDOptimizer.SDOptimizer import SDOptimizer
import matplotlib as plt
import pdb

SMOKE_DATA_DIR = "data/smoke_data_full3D"  # this data was emailed around
FALSE_ALARM_DATA_DIR = "data/bike_data_full3D"
ALARM_THRESHOLD = 4e-20

SDO = SDOptimizer()

# load the smoke data
SDO.load_timestep_directory(SMOKE_DATA_DIR)
# Get the time to alarm
smoke_source = SDO.get_time_to_alarm(spherical_projection=True,
                                     alarm_threshold=ALARM_THRESHOLD)
# And the 3D points
smoke_locs = SDO.get_3D_locs()

# Load the false alarm data
SDO.load_timestep_directory(FALSE_ALARM_DATA_DIR)
# Get the time to alarm
false_source = SDO.get_time_to_alarm(spherical_projection=True,
                                     alarm_threshold=ALARM_THRESHOLD)
# And the 3D points
false_locs = SDO.get_3D_locs()

# Do an optimization run
smoke_sources = [smoke_source, false_source]
res = SDO.optimize(smoke_sources, num_detectors=2,
                   genetic=True, multiobjective=False,
                   visualize=True, type="worst_case")

locations = res.vals
# Actually do the 3D visualization
SDO.visualize_3D(smoke_locs, smoke_source,
                 final_locations=locations, label="The smoke data")
SDO.visualize_3D(false_locs, false_source,
                 final_locations=locations, label="The bike data")
