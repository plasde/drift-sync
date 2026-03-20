import numpy as np
from nav.path_finder import a_star

import logging

logging.basicConfig(level=logging.DEBUG)


class MockWind:
    """Lightweight wind stub for testing - no Geography needed yet"""

    def __init__(self, speed=5.0, direction_deg=220):
        from math import radians, sin, cos

        self.minx, self.maxx = 0, 50000
        self.miny, self.maxy = 0, 50000
        angle = radians(direction_deg)
        self.wind_u = speed * sin(angle)
        self.wind_v = speed * cos(angle)

    def get_vector(self, x, y):
        return self.wind_u, self.wind_v


class MockGeography:
    def __init__(self):
        self.resolution = 1000
        self.minx, self.maxx = 0, 50000
        self.miny, self.maxy = 0, 50000
        size = 51
        self.sea_mask = np.ones((size, size), dtype=bool)

    def meters_to_index(self, x, y):
        return int(x / self.resolution), int(y / self.resolution)

    def index_to_meters(self, ix, iy):
        return ix * self.resolution, iy * self.resolution


geo = MockGeography()
wind = MockWind(speed=5.0, direction_deg=220)
start = (5000, 5000)
goal = (45000, 45000)

path = a_star(
    start,
    goal,
    wind_field=wind,
    geography=geo,
    step_size=5000,
)

print(f"Path found: {len(path)} waypoints")
