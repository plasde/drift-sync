import numpy as np
import heapq


# A* Pathfinding Implementation
def a_star(start, goal, obstacles):
    # Define the grid
    grid_size = 20  # Define the map size (grid 20x20)
    grid = np.zeros((grid_size, grid_size))
    
    # Mark obstacles
    for obs in obstacles:
        x, y = obs
        grid[x, y] = 1  # 1 means obstacle
    
    # A* algorithm variables
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx, ny] == 0:
                yield (nx, ny)

    # Initialize open and closed sets
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
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
            return path[::-1]
        
        for neighbor in neighbors(current):
            tentative_g_score = current_g + 1  # Assuming grid step is 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
    
    return []  # No path found