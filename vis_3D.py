import numpy as np
import SDOptimizer as SDOmodule
from SDOptimizer.SDOptimizer import SDOptimizer
import matplotlib as plt

SMOKE_DATA_DIR = "data/smoke_data_full3D"  # this data was emailed around
FALSE_ALARM_DATA_DIR = "data/bike_data_full3D"
INFEASIBLE = [(3, -2, 4, -1)]  # The infeasible region

SDO = SDOptimizer()
SDO.load_timestep_directory(SMOKE_DATA_DIR)
SDO.get_time_to_alarm()
SDO.visualize_3D()
