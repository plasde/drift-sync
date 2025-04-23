import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class SimPlotter:
    def __init__(self, boat, wind_field, target_position, obstacles, sailing_path=None):
        self.boat = boat
        self.wind_field = wind_field

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, wind_field.width)
        self.ax.set_ylim(0, wind_field.height)
        self.ax.set_aspect('equal')
        self.ax.set_title("Sailing Sim")

        self.sailing_path = sailing_path or []
        self.boat_marker, = self.ax.plot([], [], 'bo', markersize=8)
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.wind_arrow = None
        self.boat_speed_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.wind_speed_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes)


        # Plot target point (destination)
        self.ax.plot(target_position[0], target_position[1], 'gx', markersize=10, label="Goal")

        # Plot obstacles
        for ox, oy in obstacles:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black'))

        # Plot full wind field as quiver
        X, Y = np.meshgrid(np.arange(self.wind_field.width), np.arange(self.wind_field.height))
        U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)
        for y in range(self.wind_field.height):
            for x in range(self.wind_field.width):
                u, v = self.wind_field.get_vector(x, y)
                U[y, x], V[y, x] = u, v
        self.ax.quiver(X, Y, U, V, angles='xy', scale=10, color='grey', alpha=0.5)

        if self.sailing_path:
            xs = [p[0] for p in self.sailing_path]
            ys = [p[1] for p in self.sailing_path]
            self.ax.plot(xs, ys, 'k--', label="Sailing Path")
            self.ax.scatter(xs, ys, s=5, c='k')

    def init(self):
        self.boat_marker.set_data([], [])
        self.path_line.set_data([], [])
        return self.boat_marker, self.path_line

    def update(self, _):
        x, y = int(round(self.boat.pos[0])), int(round(self.boat.pos[1]))
        x = np.clip(x, 0, self.wind_field.width - 1)
        y = np.clip(y, 0, self.wind_field.height - 1)

        wind_vec = np.array(self.wind_field.get_vector(x, y))
        self.boat.update(wind_vec)

        bx, by = self.boat.pos
        self.boat_marker.set_data([bx], [by])

        history = np.array(self.boat.history)
        self.path_line.set_data(history[:, 0], history[:, 1])

        self.boat_speed_text.set_text(f"Boat speed: {self.boat.current_speed: .2f} knots")
        self.wind_speed_text.set_text(f"Wind speed: {self.wind_field.wind_speed: .2f} knots")
        
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
        plt.legend()
        plt.grid(True)
        plt.show()
