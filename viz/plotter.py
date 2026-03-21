import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import degrees


class Plotter:
    """
    Modular plotter for real-world sailing simulation
    Integrates Boat, Path, Wind, and Geography
    """

    def __init__(
        self, boat, path, wind, geography, target_position=None, obstacles=None
    ):
        self.boat = boat
        self.path = path
        self.wind = wind
        self.geo = geography
        self.target_position = target_position
        self.obstacles = obstacles or []

        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        self._setup_static_elements()
        self._setup_dynamic_elements()
        # self._plot_wind_field()

    def _setup_static_elements(self):
        # Bounds + padding
        minx, maxx, miny, maxy = self.geo.bounds_m
        pad_x = (maxx - minx) * 0.1
        pad_y = (maxy - miny) * 0.1
        self.ax.set_xlim(minx - pad_x, maxx + pad_x)
        self.ax.set_ylim(miny - pad_y, maxy + pad_y)
        self.ax.set_aspect("equal")
        self.ax.set_title("Real-World Sailing Simulation")

        # Sea mask
        self.ax.imshow(
            self.geo.sea_mask,
            origin="lower",
            cmap="Blues",
            alpha=0.3,
            extent=(minx, maxx, miny, maxy),
        )

        # Coastlines
        self.geo.coastlines_m.plot(ax=self.ax, color="black", linewidth=0.5, alpha=0.7)
        # Obstacles (if any)
        for ox, oy in self.obstacles:
            self.ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color="black"))

        # Planned path
        if self.path and len(self.path) > 1:
            coords = np.array(self.path)
            self.ax.plot(
                coords[:, 0], coords[:, 1], "k--", alpha=0.8, label="Planned Path"
            )
            self.ax.scatter(coords[:, 0], coords[:, 1], s=5, c="k")

        # Start/goal
        self.ax.plot(
            self.boat.pos[0], self.boat.pos[1], "go", markersize=10, label="Start"
        )
        self.ax.plot(
            self.target_position[0],
            self.target_position[1],
            "rx",
            markersize=12,
            label="Goal",
        )

    def _setup_dynamic_elements(self):
        # Boat marker and track
        (self.boat_marker,) = self.ax.plot([], [], "bo", markersize=8, label="Boat")
        (self.track_line,) = self.ax.plot([], [], "b-", linewidth=2, alpha=0.7)

        # Text displays
        self.speed_text = self.ax.text(
            0.02,
            0.90,
            "",
            transform=self.ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        self.wind_text = self.ax.text(
            0.02,
            0.75,
            "",
            transform=self.ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3)

    #    def _plot_wind_field(self):
    #        # Subsample grid for wind arrows
    #        X, Y = np.meshgrid(self.geo.xx[::10], self.geo.yy[::10])
    #        U, V = np.full_like(X, self.wind.wind_u), np.full_like(Y, self.wind.wind_v)
    #        self.ax.quiver(X, Y, U, V, angles="xy", color="blue", alpha=0.6, width=0.003)

    def _update_frame(self, frame):
        # Get wind at boat position
        wind_vec = self.wind.get_vector(self.boat.pos[0], self.boat.pos[1])
        self.boat.update(wind_vec)

        # Update boat marker
        self.boat_marker.set_data([self.boat.pos[0]], [self.boat.pos[1]])

        # Update path line
        if len(self.boat.history) > 1:
            hist = np.array(self.boat.history)
            self.track_line.set_data(hist[:, 0], hist[:, 1])

        # Update text
        self.speed_text.set_text(f"Speed: {self.boat.current_speed:.2f} knots")
        wind_speed = np.linalg.norm(wind_vec)
        wind_dir = (degrees(np.arctan2(wind_vec[0], wind_vec[1])) + 180) % 360
        self.wind_text.set_text(f"Wind: {wind_speed:.1f} knots from {wind_dir:.0f}°")

        return self.boat_marker, self.track_line

    def run(self, sim_duration, dt):
        frames = int(sim_duration / dt)
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=frames,
            interval=50,
            blit=False,
            repeat=False,
        )
        plt.show()
