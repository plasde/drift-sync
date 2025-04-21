import numpy as np
import matplotlib.pyplot as plt
from data.environment_data import *
import matplotlib.animation as animation
from data.environment_data import OBSTACLES, TARGET_POS

class SimPlotter:
    def __init__(self, boat, wind_vector_func):
        self.boat = boat
        self.wind_vector_func = wind_vector_func

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.ax.set_aspect('equal')
        self.ax.set_title("Sailing Sim")

        self.boat_marker, = self.ax.plot([], [], 'bo', markersize=8)
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.wind_arrow = None
        self.speed_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)


        # Plot target point (destination)
        self.ax.plot(TARGET_POS[0], TARGET_POS[1], 'gx', markersize=10, label="Goal")

        # Plot obstacles
        for ox, oy in OBSTACLES:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black'))


    def init(self):
        self.boat_marker.set_data([], [])
        self.path_line.set_data([], [])
        return self.boat_marker, self.path_line

    def update(self, _):
        self.boat.update(self.wind_vector_func())

        x, y = self.boat.pos
        self.boat_marker.set_data([x], [y])

        history = np.array(self.boat.history)
        self.path_line.set_data(history[:, 0], history[:, 1])

        # Update wind arrow
        if self.wind_arrow:
            self.wind_arrow.remove()
        wx, wy = self.wind_vector_func() / 3
        self.wind_arrow = self.ax.arrow(x, y, wx, wy, head_width=0.3, head_length=0.3, color='red')

        self.speed_text.set_text(f"Boat speed: {self.boat.current_speed: .2f} knots")

        return self.boat_marker, self.path_line

    def run(self, sim_duration, dt):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(0, int(sim_duration / dt)),
            init_func=self.init,
            blit=False,
            interval=50,
            repeat=False
        )
        plt.show()
