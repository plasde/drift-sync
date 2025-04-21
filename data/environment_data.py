from math import radians, cos, sin
import numpy as np

# Simulation parameters
WIND_DIRECTION = radians(45)      # Global wind from NE
WIND_SPEED = 1.0                  # knots
DT = 1.0                          # Time step in seconds
SIM_DURATION = 1000                # Seconds
BOAT_START_POS = np.array([10.0, 10.0])
BOAT_START_HEADING = radians(90) # East
DRAG_COEFFICIENT = 0.10
MAX_TURN_RATE = 0.200
TURN_SPEED_FACTOR = 5.0
ACCELERATION_FACTOR = 0.15

TARGET_POS = np.array([15.0, 15.0])

OBSTACLES = [(12, 12), (13, 12), (14, 12), (11, 14), (11, 15)]


def get_wind_vector(direction_degrees: float, speed_knots: float) -> np.ndarray:
    """
    Returns the wind vector given a direction in degrees (from which the wind blows)
    and speed in knots.
    """
    direction_rad = radians(direction_degrees)
    return np.array([
        speed_knots * cos(direction_rad),
        speed_knots * sin(direction_rad)
    ])
