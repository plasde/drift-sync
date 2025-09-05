import numpy as np
import requests
import matplotlib.pyplot as plt
from math import radians, degrees, sin, cos, sqrt, atan2
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
import geopandas as gpd
from pyproj import Transformer
import warnings

warnings.filterwarnings("ignore")


class RealWorldGeography:
    """
    Handles real-world geography data and coordinate transformations
    """

    def __init__(self, bbox, resolution_meters=100):
        """
        Args:
            bbox: (north, south, east, west) in degrees
            resolution_meters: Grid resolution in meters
        """
        self.bbox = bbox
        self.north, self.south, self.east, self.west = bbox
        self.resolution = resolution_meters

        # Set up coordinate transformation (WGS84 to UTM)
        self.setup_projection()

        # Convert geographic bounds to meters
        self.bounds_m = self.geo_to_meters_bounds(bbox)

        # Create grid
        self.create_grid()

        # Generate coastline and sea mask (simplified for now)
        self.generate_geography()

    def setup_projection(self):
        """Set up coordinate transformation based on area center"""
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2

        # Determine UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            epsg_code = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
        else:
            epsg_code = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere

        self.transformer = Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)
        self.inv_transformer = Transformer.from_crs(
            epsg_code, "EPSG:4326", always_xy=True
        )

    def geo_to_meters_bounds(self, bbox):
        """Convert geographic bounding box to meters"""
        north, south, east, west = bbox

        # Transform corners
        sw_x, sw_y = self.transformer.transform(west, south)
        ne_x, ne_y = self.transformer.transform(east, north)

        return (sw_x, ne_x, sw_y, ne_y)  # (minx, maxx, miny, maxy)

    def create_grid(self):
        """Create the meter-based grid"""
        print(self.bounds_m)
        minx, maxx, miny, maxy = self.bounds_m

        self.width = int((maxx - minx) / self.resolution) + 1
        self.height = int((maxy - miny) / self.resolution) + 1

        # Grid coordinates in meters
        self.x_coords = np.linspace(minx, maxx, self.width)
        self.y_coords = np.linspace(miny, maxy, self.height)

        print(f"Grid created: {self.width} x {self.height} points")
        print(f"Bounds (m): {minx:.0f} to {maxx:.0f} x, {miny:.0f} to {maxy:.0f} y")

    def generate_geography(self):
        """Generate simplified coastline and sea mask"""
        # For now, create a simple rectangular sea area with some obstacles
        # In a full implementation, you'd load real coastline data here

        minx, maxx, miny, maxy = self.bounds_m

        # Create sea mask (1 = sea, 0 = land)
        self.sea_mask = np.ones((self.height, self.width))

        # Add some simple "islands" as obstacles
        center_x, center_y = (maxx + minx) / 2, (maxy + miny) / 2

        # Create a few circular obstacles
        obstacles = [
            (center_x - 2000, center_y + 1000, 500),  # (x, y, radius)
            (center_x + 1500, center_y - 800, 300),
            (center_x - 1000, center_y - 1500, 400),
        ]

        for obs_x, obs_y, radius in obstacles:
            for i, y in enumerate(self.y_coords):
                for j, x in enumerate(self.x_coords):
                    if sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2) < radius:
                        self.sea_mask[i, j] = 0

        # Create simple coastlines (just the obstacle boundaries for now)
        self.coastlines_m = []
        for obs_x, obs_y, radius in obstacles:
            # Create circular coastline
            angles = np.linspace(0, 2 * np.pi, 50)
            coast_x = obs_x + radius * np.cos(angles)
            coast_y = obs_y + radius * np.sin(angles)
            self.coastlines_m.append(LineString(zip(coast_x, coast_y)))

    def meters_to_geo(self, x, y):
        """Convert meter coordinates back to lat/lon"""
        return self.inv_transformer.transform(x, y)

    def geo_to_meters(self, lon, lat):
        """Convert lat/lon to meter coordinates"""
        return self.transformer.transform(lon, lat)


class UniformWindField:
    """
    Uniform wind field over real-world coordinates
    """

    def __init__(self, geography, wind_speed=8.0, wind_direction=radians(220)):
        """
        Args:
            geography: RealWorldGeography object
            wind_speed: Wind speed in knots
            wind_direction: Wind direction in radians (from north, clockwise)
        """
        self.geo = geography
        self.minx, self.maxx, self.miny, self.maxy = geography.bounds_m
        self.dx = self.dy = geography.resolution
        self.width = geography.width
        self.height = geography.height

        # Uniform wind parameters
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        # Pre-calculate wind components
        self.wind_u = wind_speed * sin(wind_direction)  # Eastward component
        self.wind_v = wind_speed * cos(wind_direction)  # Northward component

        print(f"Uniform wind field created:")
        print(f"Speed: {wind_speed:.1f} knots")
        print(f"Direction: {degrees(wind_direction):.1f}° from north")

    def get_vector(self, x, y):
        """
        Get wind vector at position (x, y) in meters
        Returns (u, v) components in knots
        """
        return self.wind_u, self.wind_v

    def get_speed_direction(self, x, y):
        """Get wind speed and direction at position"""
        return self.wind_speed, self.wind_direction


class RealWorldSailingEnvironment:
    """
    Complete sailing environment with real-world coordinates
    """

    def __init__(self, bounds=(0, 0, 0, 0), resolution_meters=100):
        """
        Args:
            sailing_area: Predefined area or custom bbox
            resolution_meters: Grid resolution
        """
        self.bbox = bounds

        # Create geography and wind field
        print(f"Creating sailing environment for {self.bbox}...")
        self.geography = RealWorldGeography(self.bbox, resolution_meters)

        # Create uniform wind field (can be changed later)
        self.wind_field = UniformWindField(
            self.geography, wind_speed=8.0, wind_direction=radians(220)
        )

        print("Real-world sailing environment ready!")

    def set_wind(self, speed_knots, direction_degrees):
        """Update wind conditions"""
        direction_rad = radians(direction_degrees)
        self.wind_field = UniformWindField(
            self.geography, wind_speed=speed_knots, wind_direction=direction_rad
        )
        print(f"Wind updated: {speed_knots:.1f} knots at {direction_degrees:.1f}°")

    def get_start_goal_positions(self, start_geo, goal_geo):
        """
        Convert geographic start/goal to meter coordinates

        Args:
            start_geo: (lat, lon) in degrees
            goal_geo: (lat, lon) in degrees

        Returns:
            start_m, goal_m: Positions in meter coordinates
        """
        start_x, start_y = self.geography.geo_to_meters(start_geo[1], start_geo[0])
        goal_x, goal_y = self.geography.geo_to_meters(goal_geo[1], goal_geo[0])

        return np.array([start_x, start_y]), np.array([goal_x, goal_y])

    def plot_environment(self):
        """Plot the sailing environment"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot sea mask
        extent = [
            self.geography.bounds_m[0],
            self.geography.bounds_m[1],
            self.geography.bounds_m[2],
            self.geography.bounds_m[3],
        ]

        ax.imshow(
            self.geography.sea_mask,
            origin="lower",
            cmap="Blues",
            alpha=0.3,
            aspect="equal",
        )

        # Plot coastlines
        for coastline in self.geography.coastlines_m:
            x_coords, y_coords = coastline.xy
            ax.plot(x_coords, y_coords, "k-", linewidth=2)

        # Plot wind field
        X, Y = np.meshgrid(
            self.geography.x_coords[::5],  # Subsample for clarity
            self.geography.y_coords[::5],
        )
        U, V = (
            np.full_like(X, self.wind_field.wind_u),
            np.full_like(Y, self.wind_field.wind_v),
        )

        ax.quiver(
            X, Y, U, V, angles="xy", scale=50, color="red", alpha=0.7, width=0.003
        )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(
            f"Sailing Environment - Wind: {self.wind_field.wind_speed:.1f} knots @ {degrees(self.wind_field.wind_direction):.1f}°"
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

        return fig, ax

    # Example usage and integration with your existing code


def create_sailing_environment(
    bounds=(50.0, 52.0, 1.5, 0.0), wind_speed=8.0, wind_direction=220
):
    """
    Create a complete sailing environment

    Args:
        area: Sailing area name or (north, south, east, west) bbox
        wind_speed: Wind speed in knots
        wind_direction: Wind direction in degrees from north

    Returns:
        Environment object compatible with your existing code
    """

    # Create environment
    env = RealWorldSailingEnvironment(bounds, resolution_meters=50)

    # Set wind conditions
    env.set_wind(wind_speed, wind_direction)

    return env


# Integration class to work with your existing WindField interface
class WindFieldAdapter:
    """
    Adapter to make RealWorldSailingEnvironment compatible with your existing WindField interface
    """

    def __init__(self, sailing_env):
        self.env = sailing_env
        self.minx, self.maxx, self.miny, self.maxy = sailing_env.geography.bounds_m
        self.dx = self.dy = sailing_env.geography.resolution
        self.width = sailing_env.geography.width
        self.height = sailing_env.geography.height
        self.wind_speed = sailing_env.wind_field.wind_speed

        # For compatibility with your existing code
        self.speed_grid = np.full((self.height, self.width), self.wind_speed)
        self.dir_grid = np.full(
            (self.height, self.width), sailing_env.wind_field.wind_direction
        )

    def get_vector(self, x, y):
        """Get wind vector at position (x, y)"""
        return self.env.wind_field.get_vector(x, y)

    def generate_field(self, wind_speed, wind_direction):
        """Update wind field"""
        self.env.set_wind(
            wind_speed,
            degrees(wind_direction) if wind_direction < 10 else wind_direction,
        )
        self.wind_speed = wind_speed
        self.speed_grid.fill(wind_speed)
        self.dir_grid.fill(wind_direction)


if __name__ == "__main__":
    # Test the real-world sailing environment
    bounds = (52.0, 50.0, 1.0, 1.5)
    # Create environment
    env = create_sailing_environment(bounds=bounds, wind_speed=10, wind_direction=220)

    # Plot it
    env.plot_environment()

    start_m, goal_m = env.get_start_goal_positions(
        (bounds[0], bounds[2]), (bounds[1], bounds[3])
    )
    print(f"Dover (meters): {start_m}")
    print(f"Calais (meters): {goal_m}")

    # Create adapter for your existing code
    wind_field_adapter = WindFieldAdapter(env)
    print(
        f"Wind field adapter created with bounds: {wind_field_adapter.minx:.0f} to {wind_field_adapter.maxx:.0f}"
    )
