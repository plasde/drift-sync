import matplotlib

matplotlib.use("TkAgg")
import logging
from viz.plotter import Plotter
from core.sailboat import Sailboat
from core.wind import SpatialWind
from core.geography import Geography
from math import pi
from nav.path_finder import a_star
from data_scraper.weather import fetch_wind_grid
# from nav.rudder_control_env import RudderControlEnv
# from nav.basic_controller import BasicControllerAgent

matplotlib.set_loglevel("WARNING")
# Set up logging
logging.basicConfig(
    filename="pathfinder.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sailing_pathfinder")


# Constants for simulation
SIM_DURATION = 500
DT = 1.0

# Wind conditions
WIND_SPEED = 5.0  # knots
WIND_DIRECTION = 270  # degrees from north
WIND_GRID_RESOLUTION_DEG = 0.5  # FOr trans-Atlantic we will need to think about scaling
WIND_FORECAST_HOURS = 48
USE_HISTORICAL = False
HISTORICAL_DATE = "2024-11-15"

# SAILING_AREAS
#    "english_channel": (51.10, 51.0, 1.6, 0.0),  # Dover-Calais area
#    "san_francisco_bay": (37.9, 37.7, -122.3, -122.5),  # SF Bay
#    "mediterranean": (43.5, 42.5, 7.5, 6.0),  # Nice-Monaco area
#    "chesapeake_bay": (39.5, 37.0, -75.5, -77.0),  # Chesapeake Bay
#    "solent": (50.8, 50.7, -1.2, -1.6),  # Solent, UK

# Geographic waypoints (lat, lon) -- NOTE: so this needs to be converted to Meters
start = (52.3, 4.9)  # Amsterdam
goal = (51.5, 1.4)  # Thames estuary


if __name__ == "__main__":
    print("Setting up real-world sailing simulation...")

    # Get start and goal positions in meter coordinates
    start_pos = (start[0], start[1])
    goal_pos = (goal[0], goal[1])
    print(f"Start position: {start_pos}")
    print(f"Goal position: {goal_pos}")

    geography = Geography(start_pos=start_pos, goal_pos=goal_pos)
    start_m = geography.geo_to_meters(*start_pos[::-1])
    goal_m = geography.geo_to_meters(*goal_pos[::-1])
    simple_path = [start_m, goal_m]  # Direct line - replace with actual pathfinding
    # Debug sea mask
    sea_cells = geography.sea_mask.sum()
    total_cells = geography.sea_mask.size
    print(
        f"Sea cells: {sea_cells} / {total_cells} ({100 * sea_cells / total_cells:.1f}%)"
    )
    print(f"Geography bounds in meters: {geography.bounds_m}")

    # Test a midpoint between start and goal
    mid_m = ((start_m[0] + goal_m[0]) / 2, (start_m[1] + goal_m[1]) / 2)
    print(f"Midpoint is sea: {geography.is_sea(*mid_m)}")

    print(f" Start before snapping to sea is: {start_m}")
    print(f" Goal before snapping to sea is : {goal_m}")

    start_m = geography.snap_to_sea(*start_m)
    goal_m = geography.snap_to_sea(*goal_m)

    print(f"Start snapped to : {start_m}")
    print(f"Goal snapped to : {goal_m}")
    print(f"Start within bounds: {geography.minx <= start_m[0] <= geography.maxx}")
    print(f"Goal within bounds: {geography.minx <= goal_m[0] <= geography.maxx}")
    print(f"Start is sea: {geography.is_sea(*start_m)}")
    print(f"Goal is sea: {geography.is_sea(*goal_m)}")

    bounds = (
        geography.north,
        geography.south,
        geography.east,
        geography.west,
    )

    wind_data = fetch_wind_grid(
        bounds,
        grid_resolution_deg=WIND_GRID_RESOLUTION_DEG,
        forecast_hours=WIND_FORECAST_HOURS,
        use_historical=USE_HISTORICAL,
        historical_date=HISTORICAL_DATE if USE_HISTORICAL else None,
    )
    wind = SpatialWind(wind_data, geography)

    # Print some info
    print("\nEnvironment Info:")
    print(f" Wind: {wind.wind_speed:.1f} knots avg (spatial, 48h mean)")

    print("\nRunning A* pathfinding...")
    path = a_star(
        start=start_m,
        goal=goal_m,
        wind_field=wind,
        geography=geography,
        step_size=10000,  # Meters
        angle_resolution=pi / 16,
        course_break_penalty=1.0,
    )
    print(f"Path found: {len(path)} waypoints")
    # boat.set_path(path)

    # Init all classes
    boat = Sailboat(pos=start_m, heading=0.0, boat_type="boat1", dt=DT)

    # Create and run plotter
    plotter = Plotter(
        boat=boat,
        wind=wind,
        geography=geography,
        path=path,  # Replace with 'path' when a_star works
        target_position=goal_m,
    )

    print("\nStarting simulation...")
    plotter.run(SIM_DURATION, DT)
