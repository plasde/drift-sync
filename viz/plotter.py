import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point
from geography.geo import Geo


class SimPlotter:
    def __init__(
        self, boat, wind_field, target_position, obstacles, geo=None, sailing_path=None
    ):
        self.boat = boat
        self.wind_field = wind_field
        self.target_position = target_position
        self.coastline = geo.coastlines_m
        self.obstacles = obstacles
        self.mask = geo.sea_mask

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(wind_field.minx, wind_field.maxx)
        self.ax.set_ylim(wind_field.miny, wind_field.maxy)
        self.ax.set_aspect("equal")
        self.ax.set_title("Sailing Sim: Real world coordinates")

        self.sailing_path = sailing_path or []
        (self.boat_marker,) = self.ax.plot([], [], "bo", markersize=8)
        (self.path_line,) = self.ax.plot([], [], "b-", linewidth=1)
        self.boat_speed_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)
        self.wind_speed_text = self.ax.text(0.02, 0.90, "", transform=self.ax.transAxes)

        # Sea mask background
        self.ax.imshow(
            self.mask,
            extent=(wind_field.minx, wind_field.maxx, wind_field.miny, wind_field.maxy),
            origin="lower",
            cmap="Blues",
            alpha=0.5,
        )

        # Plot coastlines
        self.coastline.plot(ax=self.ax, color="black")

        # Plot starting point (boat's initial position)
        self.ax.plot(
            self.boat.pos[0], self.boat.pos[1], "go", markersize=10, label="Start"
        )
        # Plot target point (destination)
        self.ax.plot(
            target_position[0], target_position[1], "rx", markersize=10, label="Goal"
        )

        # Plot obstacles
        for ox, oy in obstacles:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="black"))

        # Plot full wind field as quiver
        X, Y = np.meshgrid(
            np.arange(self.wind_field.minx, self.wind_field.maxx, wind_field.dx * 5),
            np.arange(self.wind_field.miny, self.wind_field.maxy, wind_field.dy * 5),
        )
        U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)
        for i in range(Y.shape[0]):
            for j in range(X.shape[1]):
                u, v = self.wind_field.get_vector(X[0, j], Y[i, 0])
                U[i, j], V[i, j] = u, v
        self.ax.quiver(X, Y, U, V, angles="xy", scale=10, color="grey", alpha=0.5)

        if self.sailing_path:
            xs = [p[0] for p in self.sailing_path]
            ys = [p[1] for p in self.sailing_path]
            self.ax.plot(xs, ys, "k--", label="Sailing Path")
            self.ax.scatter(xs, ys, s=5, c="k")

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

        self.boat_marker.set_data([x], [y])
        history = np.array(self.boat.history)
        self.path_line.set_data(history[:, 0], history[:, 1])
        self.boat_speed_text.set_text(
            f"Boat speed: {self.boat.current_speed: .2f} knots"
        )
        self.wind_speed_text.set_text(
            f"Wind speed: {self.wind_field.wind_speed: .2f} knots"
        )

        return self.boat_marker, self.path_line

    def run(self, sim_duration, dt):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(0, int(sim_duration / dt)),
            init_func=self.init,
            blit=False,
            interval=50,
            repeat=False,
        )
        self.ax.legend(loc="upper right")
        self.ax.grid(True)
        plt.show()
