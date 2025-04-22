import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import random
from math import radians

from core.sailboat import Sailboat
from core.wind import WindField
#from data.environment_data import *
from nav.path_finder import a_star, rl_plan
from viz.plotter import *

# Constants
MAP_SIZE = [20, 20]
BOAT_START_HEADING = 0.0
BOAT_START_POS = np.array([5.0, 5.0])
TARGET_POS = np.array([17.0, 17.0])
SIM_DURATION = 100
DT = 0.1

WIND_DIRECTION = radians(random.uniform(0, 360))
WIND_SPEED = random.uniform(0.5, 7.5)

OBSTACLES = [] # [(12, 12), (13, 12), (14, 12), (11, 14), (11, 15)]


if __name__ == "__main__":
    boat = Sailboat(BOAT_START_POS.copy(), BOAT_START_HEADING, boat_type="boat1", dt = DT)
    wind_field = WindField(width=MAP_SIZE[0], height=MAP_SIZE[1])
    wind_field.generate_field(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION)

    # A* Path
    path = a_star(tuple(BOAT_START_POS.astype(int)), tuple(TARGET_POS.astype(int)), wind_field)
    boat.set_path(path)

    #wind_field.plot(boat_pos=boat.pos, goal_pos=TARGET_POS)

    plotter = SimPlotter(boat, wind_field, target_position=TARGET_POS, obstacles=OBSTACLES)
    plotter.run(SIM_DURATION, DT)
