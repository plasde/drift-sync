import numpy as np
from math import atan2, degrees, radians, cos, sin, pi
from core.polar import polar_performance

def compute_sailing_path(start, goal, wind_field, step_size=1.0, angle_resolution=15, tacking_penalty=4):
    """
    Compute a sailing path that minimizes travel time using polar performance.
    A tacking penalty is added when switching tack sides.
    Returns a list of np.array waypoints.
    """
    start = np.array(start, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)
    current = start.copy()
    path = [current.copy()]

    max_steps = 1000
    prev_tack_side = None

    for _ in range(max_steps):
        if np.linalg.norm(goal - current) < step_size:
            break

        x = int(np.clip(current[0], 0, wind_field.width - 1))
        y = int(np.clip(current[1], 0, wind_field.height - 1))
        wind_vec = wind_field.get_vector(x, y)

        wind_angle = atan2(wind_vec[1], wind_vec[0])
        vec_to_goal = goal - current

        best_score = -np.inf
        best_heading = None

        for angle_offset in range(-90, 91, angle_resolution):
            heading = atan2(vec_to_goal[1], vec_to_goal[0]) + radians(angle_offset)
            apparent_angle = (heading - wind_angle + pi) % (2 * pi) - pi

            speed = polar_performance(apparent_angle)
            dx = cos(heading) * speed
            dy = sin(heading) * speed
            progress = np.dot([dx, dy], vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-8))  # normalize for unit projection

            # Determine tack side
            tack_side = np.sign(cos(heading - wind_angle))
            if prev_tack_side is not None and tack_side != prev_tack_side:
                progress -= tacking_penalty  # apply penalty on tack switch

            if progress > best_score:
                best_score = progress
                best_heading = heading
                best_tack_side = tack_side

        if best_heading is None:
            break  # deadlock

        dx = cos(best_heading) * step_size
        dy = sin(best_heading) * step_size
        current = current + np.array([dx, dy])
        path.append(current.copy())
        prev_tack_side = best_tack_side

    path.append(goal)
    return path