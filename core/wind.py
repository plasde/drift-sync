import numpy as np
import matplotlib.pyplot as plt

class WindField:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.speed_grid = np.zeros((height, width))
        self.dir_grid = np.zeros((height, width))
        self.wind_speed = 0.0

    def generate_field(self, wind_speed, wind_direction):
        # Uniform wind direction (e.g., from 45 degrees / NE)
        self.wind_speed = wind_speed
        self.dir_grid.fill(wind_direction)
        self.speed_grid.fill(wind_speed)

    def get_vector(self, x, y):
        """Returns wind vector (u, v) at position x, y"""
        angle = self.dir_grid[y, x]
        speed = self.speed_grid[y, x]
        u = speed * np.cos(angle)
        v = speed * np.sin(angle)
        return u, v

