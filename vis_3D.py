import numpy as np
import matplotlib as plt
import pdb

import SDOptimizer as SDOmodule
from SDOptimizer.SDOptimizer import SDOptimizer
from SDOptimizer.constants import SMOKE_FOLDERS


SMOKE_DATA_DIR = "data/smoke_data_full3D"  # this data was emailed around
FALSE_ALARM_DATA_DIR = "data/bike_data_full3D"
ALARM_THRESHOLD = 4e-20
NUM_DETECTORS = 2

SDO = SDOptimizer()

all_smoke_sources = []
all_locs = []
for smoke_folder in SMOKE_FOLDERS: # Iterate over the smoke folders
    # load the smoke data
    SDO.load_timestep_directory(smoke_folder)
    # Get the time to alarm
    new_smoke_source = SDO.get_time_to_alarm(spherical_projection=True,
                                        alarm_threshold=ALARM_THRESHOLD)
    all_smoke_sources.append(new_smoke_source)
    # And the 3D points
    new_locs = SDO.get_3D_locs()
    all_locs.append(new_locs)

res = SDO.optimize(all_smoke_sources, num_detectors=NUM_DETECTORS,
                   genetic=True, multiobjective=False,
                   visualize=True, type="worst_case")

placed_detectors = res.x
# Actually do the 3D visualization

for (locs, source, label) in zip(all_locs, all_smoke_sources, SMOKE_FOLDERS):
    SDO.visualize_3D(locs, source, final_locations=placed_detectors, label=label)
