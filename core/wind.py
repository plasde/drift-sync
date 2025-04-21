from math import sin, cos
import numpy as np

# Wind vector as constant
def wind_vector():
    return np.array([
        WIND_SPEED * cos(WIND_DIRECTION),
        WIND_SPEED * sin(WIND_DIRECTION)
    ])