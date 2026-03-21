import numpy as np
from math import sin, cos


class Wind:
    """
    Updated WindField class that works with real-world meter coordinates
    and provides uniform wind across the field
    """

    def __init__(self, geography, wind_speed=8.0, wind_direction=0.0):
        """
        Parameters:
            bounds: (minx, maxx, miny, maxy) — real-world bounds in meters
            resolution: (dx, dy) — spacing between grid points in meters
            wind_speed: Uniform wind speed in knots
            wind_direction: Wind direction in radians (from north, clockwise)
        """
        self.minx, self.maxx, self.miny, self.maxy = geography.bounds_m
        self.dx = self.dy = geography.resolution

        self.width = int((self.maxx - self.minx) / self.dx) + 1
        self.height = int((self.maxy - self.miny) / self.dy) + 1

        # Wind properties
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        # Create uniform grids
        self.speed_grid = np.full((self.height, self.width), wind_speed)
        self.dir_grid = np.full((self.height, self.width), wind_direction)

        # Pre-calculate wind components for efficiency
        self.wind_u = wind_speed * sin(wind_direction)  # Eastward component
        self.wind_v = wind_speed * cos(wind_direction)  # Northward component

    def generate_field(self, wind_speed, wind_direction):
        """
        Generate uniform wind field

        Args:
            wind_speed: Wind speed in knots
            wind_direction: Wind direction in radians
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        # Update grids
        self.speed_grid.fill(wind_speed)
        self.dir_grid.fill(wind_direction)

        # Update components
        self.wind_u = wind_speed * sin(wind_direction)
        self.wind_v = wind_speed * cos(wind_direction)

    def get_vector(self, x, y):
        """
        Returns wind vector (u, v) at real-world coordinates x, y in meters

        Args:
            x, y: Position in meters

        Returns:
            (u, v): Wind components in knots (eastward, northward)
        """
        # For uniform wind, just return the constant components
        return self.wind_u, self.wind_v

    def get_wind_at_position(self, x, y):
        """
        Get wind speed and direction at real-world coordinates

        Args:
            x, y: Position in meters

        Returns:
            (speed, direction): Wind speed in knots, direction in radians
        """
        return self.wind_speed, self.wind_direction

    def position_to_grid(self, x, y):
        """
        Convert real-world coordinates to grid indices

        Args:
            x, y: Position in meters

        Returns:
            (i, j): Grid indices
        """
        j = int((x - self.minx) / self.dx)
        i = int((y - self.miny) / self.dy)

        # Clamp to grid bounds
        i = np.clip(i, 0, self.height - 1)
        j = np.clip(j, 0, self.width - 1)

        return i, j

    def grid_to_position(self, i, j):
        """
        Convert grid indices to real-world coordinates

        Args:
            i, j: Grid indices

        Returns:
            (x, y): Position in meters
        """
        x = self.minx + j * self.dx
        y = self.miny + i * self.dy
        return x, y

    def is_valid_position(self, x, y):
        """
        Check if position is within the wind field bounds

        Args:
            x, y: Position in meters

        Returns:
            bool: True if position is valid
        """
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def get_grid_coordinates(self):
        """
        Get the grid coordinate arrays

        Returns:
            (x_coords, y_coords): Arrays of grid coordinates in meters
        """
        x_coords = np.linspace(self.minx, self.maxx, self.width)
        y_coords = np.linspace(self.miny, self.maxy, self.height)
        return x_coords, y_coords

    def plot_wind_field(self, ax=None, subsample=5):
        """
        Plot the wind field as quiver plot

        Args:
            ax: Matplotlib axis (creates new if None)
            subsample: Subsample factor for arrows

        Returns:
            ax: The matplotlib axis
        """
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))

        # Get subsampled grid
        x_coords, y_coords = self.get_grid_coordinates()
        X, Y = np.meshgrid(x_coords[::subsample], y_coords[::subsample])

        # Create wind component arrays
        U = np.full_like(X, self.wind_u)
        V = np.full_like(Y, self.wind_v)

        # Plot quiver
        ax.quiver(
            X,
            Y,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            alpha=0.7,
            width=0.003,
        )

        ax.set_xlim(self.minx, self.maxx)
        ax.set_ylim(self.miny, self.maxy)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Wind Field: {self.wind_speed:.1f} knots")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        return ax

    def get_wind_at_grid(self, i, j):
        """
        Get wind at grid indices i, j

        Args:
            i, j: Grid indices

        Returns:
            (speed, direction): Wind speed and direction
        """
        i = np.clip(i, 0, self.height - 1)
        j = np.clip(j, 0, self.width - 1)

        return self.speed_grid[i, j], self.dir_grid[i, j]


class SpatialWind:
    """
    Spatially varying wind field interpolated from a fetched Open-Meteo grid.
    Drop-in replacement for Wind — same get_vector(x, y) interface.
    """

    def __init__(self, wind_data, geography):
        """
        Args:
            wind_data:  dict returned by data_scraper.weather.fetch_wind_grid()
            geography:  Geography instance (used for coordinate conversion)
        """
        self.lats = np.array(wind_data["lats"])  # ascending, shape (n_lat,)
        self.lons = np.array(wind_data["lons"])  # ascending, shape (n_lon,)
        self.u_grid = np.array(wind_data["u_grid"])  # shape (n_lat, n_lon)
        self.v_grid = np.array(wind_data["v_grid"])
        self.geography = geography

        # Summary stats (used by plotter text display)
        speeds = np.sqrt(self.u_grid**2 + self.v_grid**2)
        self.wind_speed = float(speeds.mean())

        n_lat, n_lon = self.u_grid.shape
        print(
            f"SpatialWind: {n_lat}x{n_lon} grid, "
            f"avg {self.wind_speed:.1f} kts, "
            f"lat {self.lats[0]:.1f}–{self.lats[-1]:.1f}, "
            f"lon {self.lons[0]:.1f}–{self.lons[-1]:.1f}"
        )

    def get_vector(self, x, y):
        """
        Return (u, v) wind in knots at projected meter coordinates (x, y).
        Bilinear interpolation across the lat/lon grid.
        """
        lon, lat = self.geography.meters_to_geo(x, y)
        u = self._bilinear(self.u_grid, lat, lon)
        v = self._bilinear(self.v_grid, lat, lon)
        return u, v

    def _bilinear(self, grid, lat, lon):
        """Bilinear interpolation. Clamps to grid edges rather than extrapolating."""
        lat = float(np.clip(lat, self.lats[0], self.lats[-1]))
        lon = float(np.clip(lon, self.lons[0], self.lons[-1]))

        # Lower-left corner indices
        i0 = int(np.searchsorted(self.lats, lat) - 1)
        j0 = int(np.searchsorted(self.lons, lon) - 1)
        i0 = int(np.clip(i0, 0, len(self.lats) - 2))
        j0 = int(np.clip(j0, 0, len(self.lons) - 2))
        i1, j1 = i0 + 1, j0 + 1

        # Fractional position within the cell
        lat_f = (lat - self.lats[i0]) / (self.lats[i1] - self.lats[i0])
        lon_f = (lon - self.lons[j0]) / (self.lons[j1] - self.lons[j0])

        return float(
            grid[i0, j0] * (1 - lat_f) * (1 - lon_f)
            + grid[i0, j1] * (1 - lat_f) * lon_f
            + grid[i1, j0] * lat_f * (1 - lon_f)
            + grid[i1, j1] * lat_f * lon_f
        )
