import numpy as np

class WindField:
    def __init__(self, bounds, resolution, seed=None):
        """
        Parameters:
            bounds: (minx, maxx, miny, maxy) — real-world bounds in meters
            resolution: (dx, dy) — spacing between grid points in meters
        """
        self.minx, self.maxx, self.miny, self.maxy = bounds
        self.dx, self.dy = resolution
        self.width = int((self.maxx - self.minx) / self.dx)
        self.height = int((self.maxy - self.miny) / self.dy)

        self.wind_speed = 0.0
        self.speed_grid = np.zeros((self.height, self.width))
        self.dir_grid = np.zeros((self.height, self.width))

    def generate_field(self, wind_speed, wind_direction):
        self.wind_speed = wind_speed
        self.dir_grid.fill(wind_direction)
        self.speed_grid.fill(wind_speed)

    def get_vector(self, x, y):
        """Returns interpolated wind vector (u, v) at real-world coordinates x, y"""
        ix = int((x - self.minx) / self.dx)
        iy = int((y - self.miny) / self.dy)
        ix = np.clip(ix, 0, self.width - 1)
        iy = np.clip(iy, 0, self.height - 1)

        angle = self.dir_grid[iy, ix]
        speed = self.speed_grid[iy, ix]
        u = speed * np.cos(angle)
        v = speed * np.sin(angle)
        return u, v
