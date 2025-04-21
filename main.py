import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from core.sailboat import Sailboat
from data.environment_data import *
from nav.path_finder import a_star
from viz.plotter import *

if __name__ == "__main__":
    # Init boat
    boat = Sailboat(BOAT_START_POS.copy(), BOAT_START_HEADING)
    wind_vector = lambda: get_wind_vector(WIND_DIRECTION, WIND_SPEED)

    # A* Path
    path = a_star(tuple(BOAT_START_POS.astype(int)), tuple(TARGET_POS.astype(int)), OBSTACLES)
    boat.set_path(path)

    plotter = SimPlotter(boat, wind_vector)
    plotter.run(SIM_DURATION, DT)
