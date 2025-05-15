import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import random
from math import radians
import logging

from core.sailboat import Sailboat
from core.wind import WindField
#from data.environment_data import *
from nav.path_finder import a_star 
from nav.rudder_control_env import RudderControlEnv
from nav.basic_controller import BasicControllerAgent
from viz.plotter import SimPlotter

# Set up logging
logger = logging.getLogger("sailing_pathfinder")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
      file_hander = logging.FileHandler('pathfinder.log', mode='w')
      file_hander.setLevel(logging.DEBUG)

      formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
      file_hander.setFormatter(formatter)
      
      logger.addHandler(file_hander)


# Constants
MAP_SIZE = [50, 50]
BOAT_START_HEADING = 0.0
BOAT_START_POS = np.array([10.0, 10.0])
TARGET_POS = np.array([40.0, 40.0])
SIM_DURATION = 500
DT = 1.0

WIND_DIRECTION = radians(220) #radians(random.uniform(0, 360))
WIND_SPEED = 2 #random.uniform(2.5, 7.5)

OBSTACLES = [] 

if __name__ == "__main__":
    boat = Sailboat(BOAT_START_POS.copy(), BOAT_START_HEADING, boat_type = "boat1", dt = DT)
    wind_field = WindField(width = MAP_SIZE[0], height = MAP_SIZE[1])
    wind_field.generate_field(wind_speed = WIND_SPEED, wind_direction = WIND_DIRECTION)

    path = a_star(
                start = BOAT_START_POS,
                goal = TARGET_POS,
                wind_field = wind_field,
                step_size = 1,
                course_break_penalty = 1.0,)


plotter = SimPlotter(boat,
                    wind_field,
                    target_position = TARGET_POS,
                    obstacles = OBSTACLES,
                    sailing_path = path)
plotter.run(SIM_DURATION, DT)
