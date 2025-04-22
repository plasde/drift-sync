
import numpy as np
import heapq
from core.wind import WindField
from core.polar import polar_performance
from math import atan2, pi, cos, sin

# Wind-aware A* Pathfinding
def a_star(start, goal, wind_field: WindField):
    grid_size = wind_field.width

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def neighbors(node):
        headings = np.arange(0, 2 * pi, pi / 8)  # 16 headings
        step_size = 1.0
        for angle in headings:
            dx = step_size * cos(angle)
            dy = step_size * sin(angle)
            nx, ny = node[0] + dx, node[1] + dy
            ix, iy = int(round(nx)), int(round(ny))
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                yield (ix, iy), np.array([dx, dy])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for (neighbor, movement_vec) in neighbors(current):
            if neighbor == current:
                continue

            wind = np.array(wind_field.get_vector(*neighbor))
            direction = movement_vec / np.linalg.norm(movement_vec)

            apparent_wind = wind - direction
            angle = atan2(apparent_wind[1], apparent_wind[0]) - atan2(direction[1], direction[0])
            angle = (angle + pi) % (2 * pi) - pi

            perf = polar_performance(angle)
            if perf <= 0.01:
                continue

            step_cost = 1 / perf
            tentative_g_score = current_g + step_cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

    return []  # No path found
