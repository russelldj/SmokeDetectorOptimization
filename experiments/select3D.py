import sys

import pyvista as pv

from smokedetectoroptimization.smoke_source import SmokeSource, smoke_logger
# This is a hack, but it lets us import something from the folder above.
# I will address it at some point soon.
sys.path.append("..")
from constants import SMOKE_FOLDERS, FALSE_ALARM_FOLDERS

plotter = pv.Plotter()
smoke_source = SmokeSource(SMOKE_FOLDERS[0])
plotter.add_mesh(smoke_source.XYZ, scalars=smoke_source.XYZ[:, 2])

def callback(mesh, pid):
    point = smoke_source.XYZ[pid]
    print(point)

plotter.enable_point_picking(callback=callback, show_message=True,
                       color='pink', point_size=10,
                       use_mesh=True, show_point=True)
plotter.show()
