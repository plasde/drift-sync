import numpy as np
import heapq
from core.polar import polar_performance
from math import atan2, pi, cos, sin, radians, degrees
import logging

logger = logging.getLogger("sailing_pathfinder")


# Improved Sailing-Aware A* Pathfinding
def a_star(
    start,
    goal,
    wind_field,
    geography,
    step_size=1000.0,
    angle_resolution=pi / 32,
    course_break_penalty=1.0,
    course_holding_angle=pi / 16,
):
    wind_cache = {}

    def normalize_angle(angle):
        return abs(pi - (angle) % (2 * pi))

    def heuristic(a, b, heading_set=None):
        # a and b are index tuples, convert to meters for distance
        ax, ay = geography.index_to_meters(*a)
        bx, by = geography.index_to_meters(*b)
        to_goal = np.array([bx - ax, by - ay])
        remaining_distance = np.linalg.norm(to_goal)

        wind = get_cached_wind(a)
        wind_speed = np.linalg.norm(wind)

        if heading_set is None:
            heading = atan2(wind[1], wind[0]) + radians(45)
        else:
            heading = heading_set

        wind_dir = atan2(wind[1], wind[0])
        goal_dir = atan2(to_goal[1], to_goal[0])

        tack_penalty = 0.0
        if abs(wind_dir - goal_dir) < radians(30):
            remaining_distance = remaining_distance * 0.58 * 2  # Triangle
            tack_penalty = 150.0

        wind_to_boat_angle = wind_dir - heading
        goal_to_boat_angle = goal_dir - heading

        boat_speed = polar_performance(wind_to_boat_angle) * wind_speed
        vmc = boat_speed * cos(goal_to_boat_angle)

        time_cost = remaining_distance / (vmc + 1e-2)
        return time_cost + tack_penalty

    def neighbors(node, goal_node):
        # node is index tuple, convert to meters for step calculation
        nx_m, ny_m = geography.index_to_meters(*node)
        gx_m, gy_m = geography.index_to_meters(*goal_node)

        to_goal = np.array([gx_m - nx_m, gy_m - ny_m])
        remaining_distance = np.linalg.norm(to_goal)
        angle_res_adju = (
            angle_resolution * 2
            if remaining_distance > 15 * step_size
            else angle_resolution
        )
        headings = np.arange(0, 2 * np.pi, angle_res_adju)
        for angle in headings:
            dx = step_size * cos(angle)
            dy = step_size * sin(angle)
            new_x = nx_m + dx
            new_y = ny_m + dy
            nix, niy = geography.meters_to_index(new_x, new_y)
            if (
                0 <= niy < geography.sea_mask.shape[0]
                and 0 <= nix < geography.sea_mask.shape[1]
            ):
                if geography.sea_mask[niy, nix]:
                    yield (nix, niy), np.array([dx, dy]), angle

    def snap_to_grid(pos):
        ix, iy = geography.meters_to_index(pos[0], pos[1])
        return (ix, iy)

    def index_to_meters(node):
        return geography.index_to_meters(node[0], node[1])

    def get_cached_wind(pos):
        ix, iy = pos
        x, y = geography.index_to_meters(ix, iy)
        return wind_cache.setdefault((ix, iy), np.array(wind_field.get_vector(x, y)))

    def compute_heading_metrics(wind_vec, movement_vec):
        wind_speed = np.linalg.norm(wind_vec)
        if wind_speed < 1e-3:
            return {"apparent_angle": pi / 2, "polar_perf": 0.5, "boat_speed": 0}

        wind_angle = atan2(wind_vec[1], wind_vec[0])
        heading = atan2(movement_vec[1], movement_vec[0])
        wind_to_boat_angle = normalize_angle(wind_angle - heading)

        perf = polar_performance(wind_to_boat_angle)
        boat_speed = perf * wind_speed

        return {
            "apparent_angle": wind_to_boat_angle,
            "polar_perf": perf,
            "boat_speed": boat_speed,
        }

    def evaluate_neighbor(
        neighbor,
        movement_vec,
        heading,
        current_g,
        current_tack,
        current_heading,
        wind_vec,
        goal_node,
        boat_speed,
    ):
        step_dist = np.linalg.norm(movement_vec)
        if boat_speed < 1e-3:
            return None

        # Tack logic
        wind_angle = atan2(wind_vec[1], wind_vec[0])
        angle_diff = (wind_angle - heading) % (2 * pi)
        if angle_diff > pi:
            angle_diff -= 2 * pi
        new_tack = np.sign(angle_diff)
        tack_changed = current_tack is not None and float(new_tack) != float(
            current_tack
        )
        tack_penalty = 500.0 if tack_changed else 0.0
        if tack_changed:
            boat_speed *= 0.5

        # Course penalty
        course_change_penalty = 0.0
        if current_heading is not None:
            heading_change = normalize_angle(heading - current_heading)
            if heading_change > course_holding_angle:
                course_change_penalty = heading_change * course_break_penalty * 0.5

        time_cost = step_dist / (boat_speed)
        g = current_g + time_cost + tack_penalty + course_change_penalty
        h = heuristic(neighbor, goal_node, heading_set=heading)

        return {
            "g": g,
            "h": h,
            "f": g + h,
            "metrics": metrics,
            "tack_changed": tack_changed,
            "tack_side": new_tack,
            "tack_penalty": tack_penalty,
            "course_penalty": course_change_penalty,
            "time_cost": time_cost,
        }

    def build_came_from_entry(current, heading, eval_result):
        return {
            "from": current,
            "tack_side": eval_result["tack_side"],
            "boat_speed": eval_result["metrics"]["boat_speed"],
            "heading": heading,
            "apparent_angle": eval_result["metrics"]["apparent_angle"],
            "polar_perf": eval_result["metrics"]["polar_perf"],
            "fixed_tack_penalty": eval_result["tack_penalty"],
            "course_change_penalty": eval_result["course_penalty"],
            "time_cost": eval_result["time_cost"],
            "heuristic_cost": eval_result["h"],
            "total_cost": eval_result["f"],
            "tack_changed": eval_result["tack_changed"],
            "old_tack_side": current_tack,
            "new_tack_side": eval_result["tack_side"],
        }

    ########## Initializing the A* algorithm ##########
    open_set = []
    start_node = snap_to_grid((start[0], start[1]))
    goal_node = snap_to_grid((goal[0], goal[1]))

    heapq.heappush(
        open_set, (heuristic(start_node, goal_node), 0, start_node, None, None)
    )
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        if len(g_score) > 50000:
            logger.warning("Too many nodes explored, stopping search.")
            break

        _, current_g, current, current_tack, current_heading = heapq.heappop(open_set)

        #### Checking if goal is reached ####
        dist_to_goal = (
            np.sqrt((current[0] - goal_node[0]) ** 2 + (current[1] - goal_node[1]) ** 2)
            * geography.resolution
        )
        if dist_to_goal < step_size:
            path = []
            node = current
            visited = set()
            while node in came_from:
                if node in visited:
                    logger.warning(f"Cycle detected in path reconstruction at {node}.")
                    break
                visited.add(node)
                path.append(node)
                node = came_from[node]["from"]
            path.append(start_node)
            path.reverse()

            logger.info("Path found. Explaining steps:")
            for i in range(len(path) - 1):
                step = path[i + 1]
                prev = path[i]
                step_info = came_from.get(step)
                if step_info is None:
                    logger.warning(f"No metadata for step {step}")
                    continue
                logger.debug(
                    f"Step {i + 1}: {prev} -> {step} | "
                    f"Heading: {degrees(step_info['heading']):.1f}°, "
                    f"App. Wind Angle: {degrees(step_info['apparent_angle']):.1f}°, "
                    f"Perf: {step_info['polar_perf']:.2f}, "
                    f"Speed: {step_info['boat_speed']:.2f}, "
                    f"Tack Penalty: {step_info['fixed_tack_penalty']}, "
                    f"Course Penalty: {step_info['course_change_penalty']:.2f}, "
                    f"Time: {step_info['time_cost']:.2f}, "
                    f"Heuristic: {step_info['heuristic_cost']:.2f}, "
                    f"Total f: {step_info['total_cost']:.2f},"
                    f"Tack changed: {step_info['tack_changed']}, old tack: {step_info['old_tack_side']}, new tack: {step_info['new_tack_side']}"
                )
            # Convert index path to meter coordinates
            return [index_to_meters(node) for node in path]

        #### Exploring neighbors ####
        for neighbor, movement_vec, heading in neighbors(current, goal_node):
            wind = get_cached_wind(current)
            if np.linalg.norm(wind) < 1e-3:
                continue

            metrics = compute_heading_metrics(wind, movement_vec)
            perf = metrics["polar_perf"]
            if perf < 0.3:
                continue
            boat_speed = metrics["boat_speed"]

            eval_result = evaluate_neighbor(
                neighbor=neighbor,
                movement_vec=movement_vec,
                heading=heading,
                current_g=current_g,
                current_tack=current_tack,
                current_heading=current_heading,
                wind_vec=wind,
                goal_node=goal_node,
                boat_speed=boat_speed,
            )

            if eval_result is None:
                continue

            if neighbor not in g_score or eval_result["g"] < g_score[neighbor]:
                g_score[neighbor] = eval_result["g"]
                f_score[neighbor] = eval_result["f"]
                heapq.heappush(
                    open_set,
                    (
                        eval_result["f"],
                        eval_result["g"],
                        neighbor,
                        eval_result["tack_side"],
                        heading,
                    ),
                )
                came_from[neighbor] = build_came_from_entry(
                    current, heading, eval_result
                )

    logger.warning(f"No path found. Nodes explored: {len(g_score)}")
    return []
