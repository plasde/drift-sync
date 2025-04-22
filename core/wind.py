import numpy as np
import matplotlib.pyplot as plt

class WindField:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.speed_grid = np.zeros((height, width))
        self.dir_grid = np.zeros((height, width))
        self.wind_speed = 0.0

    def generate_field(self, wind_speed, wind_direction):
        # Uniform wind direction (e.g., from 45 degrees / NE)
        self.wind_speed = wind_speed
        self.dir_grid.fill(wind_direction)
        self.speed_grid.fill(wind_speed)

    def get_vector(self, x, y):
        """Returns wind vector (u, v) at position x, y"""
        angle = self.dir_grid[y, x]
        speed = self.speed_grid[y, x]
        u = speed * np.cos(angle)
        v = speed * np.sin(angle)
        return u, v

    def plot(self, boat_pos=None, goal_pos=None):
        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)

        for y in range(self.height):
            for x in range(self.width):
                u, v = self.get_vector(x, y)
                U[y, x], V[y, x] = u, v

        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, U, V, angles='xy', scale=10, color='blue')
        plt.title("Uniform Wind Field")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal')
        #plt.gca().invert_yaxis()
        plt.grid(True)

        # Plot boat position
        if boat_pos is not None:
            plt.plot(boat_pos[0], boat_pos[1], 'ro', markersize=10, label="Boat")

        # Plot goal position
        if goal_pos is not None:
            plt.plot(goal_pos[0], goal_pos[1], 'go', markersize=10, label="Goal")

        if boat_pos or goal_pos:
            plt.legend()

        plt.show()
