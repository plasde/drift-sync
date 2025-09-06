import matplotlib

matplotlib.use("TkAgg")
import logging
from viz.plotter import Plotter
from core.sailboat import Sailboat
from core.wind import Wind
from core.geography import Geography

# from nav.path_finder import a_star
# from nav.rudder_control_env import RudderControlEnv
# from nav.basic_controller import BasicControllerAgent

matplotlib.set_loglevel("WARNING")
# Set up logging
logging.basicConfig(
    filename="pathfinder.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sailing_pathfinder")


# Constants for simulation
SIM_DURATION = 500
DT = 1.0
# Wind conditions
WIND_SPEED = 5.0  # knots
WIND_DIRECTION = 220  # degrees from north

# SAILING_AREAS
#    "english_channel": (51.10, 51.0, 1.6, 0.0),  # Dover-Calais area
#    "san_francisco_bay": (37.9, 37.7, -122.3, -122.5),  # SF Bay
#    "mediterranean": (43.5, 42.5, 7.5, 6.0),  # Nice-Monaco area
#    "chesapeake_bay": (39.5, 37.0, -75.5, -77.0),  # Chesapeake Bay
#    "solent": (50.8, 50.7, -1.2, -1.6),  # Solent, UK

# Geographic waypoints (lat, lon) -- NOTE: so this needs to be converted to Meters
start = (52.3, 4.9)
goal = (51.5, 0.0)


if __name__ == "__main__":
    print("Setting up real-world sailing simulation...")

    # Get start and goal positions in meter coordinates
    start_pos = (start[0], start[1])
    goal_pos = (goal[0], goal[1])
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")

    # Print some info
    print("\nEnvironment Info:")
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

    # Init all classes
    geography = Geography(start_pos=start_pos, goal_pos=goal_pos)
    start_m = geography.geo_to_meters(*start_pos[::-1])
    goal_m = geography.geo_to_meters(*goal_pos[::-1])
    simple_path = [start_m, goal_m]  # Direct line - replace with actual pathfinding

    boat = Sailboat(pos=start_m, heading=0.0, boat_type="boat1", dt=DT)
    wind = Wind(geography)

    # Create and run plotter
    plotter = Plotter(
        boat=boat,
        wind=wind,
        geography=geography,
        path=simple_path,  # Replace with 'path' when a_star works
        target_position=goal_m,
    )

    print("\nStarting simulation...")
    plotter.run(SIM_DURATION, DT)
