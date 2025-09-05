"""
Minimal example showing how to integrate real-world geography with your existing code
"""

import numpy as np
from math import radians, degrees
import matplotlib.pyplot as plt

# Your existing imports (adapt paths as needed)
from core.sailboat import Sailboat
from updated_wind_field import WindField  # Use the updated WindField class


def create_real_world_sailing_setup(area="english_channel"):
    """
    Create a real-world sailing setup that works with your existing code
    """

    # Define sailing areas (north, south, east, west)
    areas = {
        "english_channel": (51.2, 50.8, 1.5, -1.0),
        "san_francisco_bay": (37.9, 37.7, -122.3, -122.5),
        "solent": (50.8, 50.7, -1.2, -1.6),
    }

    # Geographic waypoints (lat, lon)
    waypoints = {
        "english_channel": {
            "start": (51.10, 1.20),  # Near Dover
            "goal": (51.00, 1.60),  # Towards Calais
        },
        "san_francisco_bay": {"start": (37.82, -122.42), "goal": (37.75, -122.38)},
        "solent": {"start": (50.75, -1.30), "goal": (50.74, -1.50)},
    }

    bbox = areas[area]
    wp = waypoints[area]

    # Simple coordinate transformation (approximate)
    # For more accuracy, use the full RealWorldGeography class
    def simple_geo_to_meters(lat, lon, ref_lat, ref_lon):
        """Simple approximation for small areas"""
        # Rough conversion: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
        x = (lon - ref_lon) * 111000 * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * 111000
        return x, y

    # Use center of area as reference
    ref_lat = (bbox[0] + bbox[1]) / 2
    ref_lon = (bbox[2] + bbox[3]) / 2

    # Convert bbox to meters
    north, south, east, west = bbox

    west_m, south_m = simple_geo_to_meters(south, west, ref_lat, ref_lon)
    east_m, north_m = simple_geo_to_meters(north, east, ref_lat, ref_lon)

    bounds_m = (west_m, east_m, south_m, north_m)  # (minx, maxx, miny, maxy)

    # Convert waypoints to meters
    start_x, start_y = simple_geo_to_meters(
        wp["start"][0], wp["start"][1], ref_lat, ref_lon
    )
    goal_x, goal_y = simple_geo_to_meters(
        wp["goal"][0], wp["goal"][1], ref_lat, ref_lon
    )

    start_pos = np.array([start_x, start_y])
    goal_pos = np.array([goal_x, goal_y])

    return bounds_m, start_pos, goal_pos, ref_lat, ref_lon


def run_sailing_simulation():
    """
    Run a basic sailing simulation with real-world coordinates
    """

    print("Setting up real-world sailing simulation...")

    # Set up the sailing area
    area = "english_channel"
    bounds_m, start_pos, goal_pos, ref_lat, ref_lon = create_real_world_sailing_setup(
        area
    )

    print(f"Sailing area: {area}")
    print(f"Bounds (m): {bounds_m}")
    print(f"Start position (m): {start_pos}")
    print(f"Goal position (m): {goal_pos}")
    print(f"Reference point: {ref_lat:.4f}°N, {ref_lon:.4f}°E")

    # Create wind field with uniform wind
    wind_speed = 8.0  # knots
    wind_direction = radians(220)  # SW wind

    wind_field = WindField(
        bounds=bounds_m,
        resolution=100,  # 100m grid spacing
        wind_speed=wind_speed,
        wind_direction=wind_direction,
    )

    print(f"Wind field created: {wind_field.width}x{wind_field.height} grid")
    print(f"Wind: {wind_speed:.1f} knots @ {degrees(wind_direction):.1f}°")

    # Create sailboat
    boat = Sailboat(pos=start_pos.copy(), heading=0.0, boat_type="boat1", dt=1.0)

    # Set simple path (direct to goal for now)
    # Later replace with: path = a_star(start_pos, goal_pos, wind_field, ...)
    simple_path = [start_pos, goal_pos]
    boat.set_path(simple_path)

    # Run basic simulation
    print("\nRunning simulation...")

    # Set up plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Wind field
    wind_field.plot_wind_field(ax1)
    ax1.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start")
    ax1.plot(goal_pos[0], goal_pos[1], "rx", markersize=10, label="Goal")
    ax1.plot(
        [start_pos[0], goal_pos[0]],
        [start_pos[1], goal_pos[1]],
        "k--",
        alpha=0.5,
        label="Direct Route",
    )
    ax1.legend()
    ax1.set_title("Wind Field and Route")

    # Simulate boat movement
    boat_positions = [start_pos.copy()]
    speeds = []

    for step in range(200):  # 200 steps
        # Get wind at current position
        wind_u, wind_v = wind_field.get_vector(boat.pos[0], boat.pos[1])
        wind_vec = np.array([wind_u, wind_v])

        # Update boat
        boat.update(wind_vec)
        boat_positions.append(boat.pos.copy())
        speeds.append(boat.current_speed)

        # Check if reached goal
        distance_to_goal = np.linalg.norm(boat.pos - goal_pos)
        if distance_to_goal < 500:  # 500m tolerance
            print(f"Reached goal in {step} steps!")
            break

    # Plot 2: Boat track and speed
    boat_positions = np.array(boat_positions)
    ax2.plot(
        boat_positions[:, 0],
        boat_positions[:, 1],
        "b-",
        linewidth=2,
        label="Boat Track",
    )
    ax2.plot(start_pos[0], start_pos[1], "go", markersize=10, label="Start")
    ax2.plot(goal_pos[0], goal_pos[1], "rx", markersize=10, label="Goal")
    ax2.plot(boat.pos[0], boat.pos[1], "bo", markersize=8, label="Current Position")

    # Add wind arrows
    x_coords, y_coords = wind_field.get_grid_coordinates()
    X, Y = np.meshgrid(x_coords[::10], y_coords[::10])
    U, V = np.full_like(X, wind_field.wind_u), np.full_like(Y, wind_field.wind_v)
    ax2.quiver(X, Y, U, V, angles="xy", scale=100, color="red", alpha=0.5, width=0.002)

    ax2.set_xlim(bounds_m[0], bounds_m[1])
    ax2.set_ylim(bounds_m[2], bounds_m[3])
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    ax2.set_title("Boat Track")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()

    # Print final stats
    total_distance = sum(
        np.linalg.norm(boat_positions[i + 1] - boat_positions[i])
        for i in range(len(boat_positions) - 1)
    )
    direct_distance = np.linalg.norm(goal_pos - start_pos)

    print(f"\nSimulation Results:")
    print(f"Total distance sailed: {total_distance:.0f}m")
    print(f"Direct distance: {direct_distance:.0f}m")
    print(f"Efficiency: {direct_distance / total_distance * 100:.1f}%")
    print(f"Average speed: {np.mean(speeds):.2f} knots")
    print(f"Final position: ({boat.pos[0]:.0f}, {boat.pos[1]:.0f})")

    plt.show()

    return wind_field, boat, boat_positions


# Integration function for your A* pathfinder
def create_wind_field_for_astar(
    area="english_channel", wind_speed=8.0, wind_direction_deg=220
):
    """
    Create a WindField that can be used with your a_star pathfinder

    Args:
        area: Sailing area name
        wind_speed: Wind speed in knots
        wind_direction_deg: Wind direction in degrees from north

    Returns:
        wind_field, start_pos, goal_pos: Ready for a_star
    """

    # Set up area
    bounds_m, start_pos, goal_pos, _, _ = create_real_world_sailing_setup(area)

    # Create wind field
    wind_field = WindField(
        bounds=bounds_m,
        resolution=50,  # 50m resolution for pathfinding
        wind_speed=wind_speed,
        wind_direction=radians(wind_direction_deg),
    )

    return wind_field, start_pos, goal_pos


if __name__ == "__main__":
    # Run the simulation
    wind_field, boat, positions = run_sailing_simulation()

    print("\n" + "=" * 50)
    print("INTEGRATION READY!")
    print("=" * 50)
    print("You can now use this with your A* pathfinder like this:")
    print("")
    print(
        "wind_field, start_pos, goal_pos = create_wind_field_for_astar('english_channel')"
    )
    print("path = a_star(")
    print("    start=start_pos,")
    print("    goal=goal_pos,")
    print("    wind_field=wind_field,")
    print("    step_size=100,  # meters")
    print("    course_break_penalty=1.0")
    print(")")
