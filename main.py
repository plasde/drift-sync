import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import random
from math import radians

from core.sailboat import Sailboat
from core.wind import WindField
#from data.environment_data import *
from nav.path_finder import a_star
from nav.sailing_path_calculator import compute_sailing_path
from nav.rudder_control_env import RudderControlEnv
from nav.basic_controller import BasicControllerAgent
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

def wind_fn(pos):
        x = int(np.clip(pos[0], 0, MAP_SIZE[0]-1))
        y = int(np.clip(pos[1], 0, MAP_SIZE[1]-1))
        return wind_field.get_vector(x, y)


if __name__ == "__main__":
    boat = Sailboat(BOAT_START_POS.copy(), BOAT_START_HEADING, boat_type="boat1", dt = DT)
    wind_field = WindField(width=MAP_SIZE[0], height=MAP_SIZE[1])
    wind_field.generate_field(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION)

    # Using path calculator
    path = compute_sailing_path(BOAT_START_POS, TARGET_POS, wind_field)
    #boat.set_path(path)

    # RL based path
    #env = RudderControlEnv(BOAT_START_POS, TARGET_POS, wind_fn=wind_fn, dt=DT)
    #agent = BasicControllerAgent()

    #obs =  env.reset()
    #done = False
    #t = 0
    #while not done:
    #    rudder_angle = 0.2 * np.sin(t * 0.05)  # No action, just for testing
    #    action = [rudder_angle]
    #    obs, reward, done, info = env.step(action)
    #    t += 1
    
    #boat = env.boat

    #wind_field.plot(boat_pos=boat.pos, goal_pos=TARGET_POS)

    plotter = SimPlotter(boat, wind_field, target_position=TARGET_POS, obstacles=OBSTACLES, sailing_path=path)
    plotter.run(SIM_DURATION, DT)
