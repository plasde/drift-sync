import numpy as np
import random
from math import radians, degrees
import logging

from core.sailboat import Sailboat
from core.wind import WindField
#from data.environment_data import *
from nav.path_finder import a_star 
#from nav.rudder_control_env import RudderControlEnv
#from nav.basic_controller import BasicControllerAgent
from viz.plotter import SimPlotter
from geography.geo import Geo

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
SIM_DURATION = 500
DT = 1.0
WIND_DIRECTION = radians(135) #radians(random.uniform(0, 360))
WIND_SPEED = random.uniform(2.5, 7.5)
START_PORT_COORD = (5.4200, 53.1735)  # Harlingen
DEST_PORT_COORD = (4.7594, 52.9563)    # Den Helder

if __name__ == "__main__":
    geo = Geo(dx=1001, dy=1000, epsg_id=28992)
    start_x, start_y = geo.snap_to_sea(*geo.transformer.transform(*START_PORT_COORD))  # Harlingen
    goal_x, goal_y = geo.snap_to_sea(*geo.transformer.transform(*DEST_PORT_COORD))  # Den Helder

    BOAT_START_POS = np.array([start_x, start_y])
    TARGET_POS = np.array([goal_x, goal_y])
    boat = Sailboat(BOAT_START_POS.copy(), heading=0.0, boat_type = "boat1", dt = DT)

    wind_field = WindField(bounds=(geo.minx, geo.maxx, geo.miny, geo.maxy), resolution=(geo.dx,geo.dy))
    wind_field.generate_field(wind_speed = WIND_SPEED, wind_direction = WIND_DIRECTION)

    print("Starting pathfinding...")
    path = a_star(
                start = BOAT_START_POS,
                goal = TARGET_POS,
                wind_field = wind_field,
                geo = geo,
                step_size = 1000,
                grid_resolution = 1000,
                course_break_penalty = 1.0,)
    print("Pathfinding complete. Path length:", len(path))
    if len(path) == 0:
        logger.error("No path found!")
        logger.error("Wind speed: %.2f knots, Wind direction: %.2f degrees", WIND_SPEED, degrees(WIND_DIRECTION))
        exit(1)


plotter = SimPlotter(boat = boat,
                    wind_field = wind_field,
                    target_position = TARGET_POS,
                    obstacles = [], # useless for now
                    geo = geo,
                    sailing_path = path)
plotter.run(SIM_DURATION, DT)
