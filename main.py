import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from math import radians, degrees
import logging

from core.sailboat import Sailboat
from core.wind import WindField
# from nav.path_finder import a_star
# from nav.rudder_control_env import RudderControlEnv
# from nav.basic_controller import BasicControllerAgent
# from viz.plotter import SimPlotter

# Import the real-world geography system
from real_world_sailing_system import create_sailing_environment, WindFieldAdapter

# Set up logging
logger = logging.getLogger("sailing_pathfinder")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    file_handler = logging.FileHandler("pathfinder.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Constants for simulation
SIM_DURATION = 500
DT = 1.0

# Wind conditions
WIND_SPEED = 8.0  # knots
WIND_DIRECTION = 220  # degrees from north

# Sailing area selection
SAILING_AREAS = {
    "english_channel": (51.2, 50.8, 1.5, -1.0),  # Dover-Calais area
    "san_francisco_bay": (37.9, 37.7, -122.3, -122.5),  # SF Bay
    "mediterranean": (43.5, 42.5, 7.5, 6.0),  # Nice-Monaco area
    "chesapeake_bay": (39.5, 37.0, -75.5, -77.0),  # Chesapeake Bay
    "solent": (50.8, 50.7, -1.2, -1.6),  # Solent, UK
}

CURRENT_AREA = "english_channel"

# Geographic waypoints (lat, lon)
WAYPOINTS = {
    "english_channel": {
        "dover": (51.13, 1.31),
        "calais": (50.95, 1.85),
        "start": (51.10, 1.20),
        "goal": (51.00, 1.60),
    },
    "san_francisco_bay": {
        "start": (37.82, -122.42),  # Near Golden Gate
        "goal": (37.75, -122.38),  # Near Bay Bridge
    },
    "solent": {
        "start": (50.75, -1.30),  # Near Portsmouth
        "goal": (50.74, -1.50),  # Near Yarmouth
    },
}

OBSTACLES = []


class GeographyPlotter:
    """
    Updated plotter that works with real-world geography
    """

    def __init__(
        self, boat, sailing_env, target_position, obstacles=None, sailing_path=None
    ):
        self.boat = boat
        self.sailing_env = sailing_env
        self.wind_field = sailing_env.wind_field
        self.geography = sailing_env.geography
        self.target_position = target_position
        self.obstacles = obstacles or []
        self.sailing_path = sailing_path

        # Set up plot
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()

    def setup_plot(self):
        """Set up the plotting area"""
        bounds = self.geography.bounds_m
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_aspect("equal")
        self.ax.set_title("Real-World Sailing Simulation")

        # Plot geography
        self.plot_geography()
        self.plot_wind_field()
        self.setup_dynamic_elements()

    def plot_geography(self):
        """Plot the geographic elements"""
        bounds = self.geography.bounds_m
        extent = [bounds[0], bounds[1], bounds[2], bounds[3]]

        # Sea mask
        self.ax.imshow(
            self.geography.sea_mask,
            extent=extent,
            origin="lower",
            cmap="Blues",
            alpha=0.3,
            aspect="equal",
        )

        # Coastlines
        for coastline in self.geography.coastlines_m:
            x_coords, y_coords = coastline.xy
            self.ax.plot(x_coords, y_coords, "k-", linewidth=2, label="Coastline")

    def plot_wind_field(self):
        """Plot wind vectors"""
        # Subsample grid for wind arrows
        X, Y = np.meshgrid(self.geography.x_coords[::10], self.geography.y_coords[::10])

        U = np.full_like(X, self.wind_field.wind_u)
        V = np.full_like(Y, self.wind_field.wind_v)

        self.ax.quiver(
            X, Y, U, V, angles="xy", scale=100, color="red", alpha=0.6, width=0.003
        )

    def setup_dynamic_elements(self):
        """Set up elements that will be updated during animation"""
        # Boat marker
        (self.boat_marker,) = self.ax.plot([], [], "bo", markersize=8, label="Boat")

        # Boat path
        (self.path_line,) = self.ax.plot(
            [], [], "b-", linewidth=2, alpha=0.7, label="Boat Track"
        )

        # Target
        self.ax.plot(
            self.target_position[0],
            self.target_position[1],
            "rx",
            markersize=12,
            label="Target",
        )

        # Start position
        self.ax.plot(
            self.boat.pos[0], self.boat.pos[1], "go", markersize=10, label="Start"
        )

        # Planned sailing path
        if self.sailing_path:
            path_array = np.array(self.sailing_path)
            self.ax.plot(
                path_array[:, 0],
                path_array[:, 1],
                "k--",
                alpha=0.8,
                label="Planned Route",
            )

        # Text displays
        self.boat_speed_text = self.ax.text(
            0.02,
            0.95,
            "",
            transform=self.ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        self.wind_text = self.ax.text(
            0.02,
            0.88,
            "",
            transform=self.ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3)

    def update_animation(self, frame):
        """Update function for animation"""
        # Get wind at current position
        wind_u, wind_v = self.wind_field.get_vector(self.boat.pos[0], self.boat.pos[1])
        wind_vec = np.array([wind_u, wind_v])

        # Update boat
        self.boat.update(wind_vec)

        # Update displays
        self.boat_marker.set_data([self.boat.pos[0]], [self.boat.pos[1]])

        if len(self.boat.history) > 1:
            history = np.array(self.boat.history)
            self.path_line.set_data(history[:, 0], history[:, 1])

        # Update text
        self.boat_speed_text.set_text(
            f"Boat Speed: {self.boat.current_speed:.2f} knots"
        )
        wind_speed = np.linalg.norm([wind_u, wind_v])
        wind_dir = degrees(np.arctan2(wind_u, wind_v))
        self.wind_text.set_text(f"Wind: {wind_speed:.1f} knots @ {wind_dir:.0f}°")

        return self.boat_marker, self.path_line

    def run(self, sim_duration, dt):
        """Run the simulation"""
        import matplotlib.animation as animation

        frames = int(sim_duration / dt)
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            frames=frames,
            interval=50,
            blit=False,
            repeat=False,
        )

        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    print("Setting up real-world sailing simulation...")

    # Create real-world sailing environment
    sailing_env = create_sailing_environment(
        area=CURRENT_AREA, wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION
    )

    # Get start and goal positions in meter coordinates
    waypoints = WAYPOINTS[CURRENT_AREA]
    start_geo = waypoints["start"]
    goal_geo = waypoints["goal"]

    start_pos, goal_pos = sailing_env.get_start_goal_positions(start_geo, goal_geo)

    print(f"Start position: {start_geo} -> {start_pos}")
    print(f"Goal position: {goal_geo} -> {goal_pos}")

    # Create boat
    boat = Sailboat(pos=start_pos.copy(), heading=0.0, boat_type="boat1", dt=DT)

    # For now, create a simple path (replace with your a_star later)
    simple_path = [start_pos, goal_pos]  # Direct line - replace with actual pathfinding
    boat.set_path(simple_path)

    # Create the adapter for compatibility with your existing WindField interface
    wind_field_adapter = WindFieldAdapter(sailing_env)

    # Print some info
    print("\nEnvironment Info:")
    print(f"Grid size: {wind_field_adapter.width} x {wind_field_adapter.height}")
    print(f"Resolution: {sailing_env.geography.resolution:.0f}m")
    print(
        f"Bounds (m): {wind_field_adapter.minx:.0f} to {wind_field_adapter.maxx:.0f} x {wind_field_adapter.miny:.0f} to {wind_field_adapter.maxy:.0f}"
    )
    print(f"Wind: {WIND_SPEED:.1f} knots @ {WIND_DIRECTION}°")

    # Uncomment this when you have a_star working:
    # print("\nRunning A* pathfinding...")
    # path = a_star(
    #     start=start_pos,
    #     goal=goal_pos,
    #     wind_field=wind_field
    #     step_size=100,  # Meters
    #     course_break_penalty=1.0
    # )
    # boat.set_path(path)

    # Create and run plotter
    plotter = GeographyPlotter(
        boat=boat,
        sailing_env=sailing_env,
        target_position=goal_pos,
        obstacles=OBSTACLES,
        sailing_path=simple_path,  # Replace with 'path' when a_star works
    )

    print("\nStarting simulation...")
    plotter.run(SIM_DURATION, DT)
