import numpy as np
from math import atan2, pi, sin, cos
from core.polar import polar_performance

class Sailboat:
    def __init__(self, pos, heading, boat_type, dt):
        self.pos = pos
        self.heading = heading
        self.history = [pos.copy()]
        self.current_speed = 0
        self.velocity = np.array([0.0, 0.0])
        self.path = []
        self.path_index = 0
        if boat_type == "boat1":
            self.drag_coefficient = 0.10
            self.max_turn_rate = 0.2
            self.turn_speed_factor = 5.0
            self.acceleration_factor = 0.15
        self.dt = dt

    def set_path(self, path):
        self.path = path
        self.path_index = 0

    def update(self, wind_vec):
        if self.path_index >= len(self.path):
            return

        self._compute_apparent_wind(wind_vec)
        self._calculate_target_angle()
        self._adjust_heading()
        self._compute_target_speed()
        self._adjust_speed()
        self._apply_drag()
        self._update_velocity()
        self._move()

    def _compute_apparent_wind(self, wind_vec):
        self.apparent_wind_vec = wind_vec - self.velocity
        self.apparent_wind_speed = np.linalg.norm(self.apparent_wind_vec)
        if self.apparent_wind_speed > 0.001:
            angle = atan2(self.apparent_wind_vec[1], self.apparent_wind_vec[0]) - self.heading
            self.apparent_wind_angle = (angle + pi) % (2 * pi) - pi
        else:
            self.apparent_wind_angle = 0

    def _calculate_target_angle(self):
        self.target = self.path[self.path_index]
        self.direction_to_target = self.target - self.pos
        self.distance_to_target = np.linalg.norm(self.direction_to_target)
        self.target_angle = atan2(self.direction_to_target[1], self.direction_to_target[0])

    def _adjust_heading(self):
        heading_diff = (self.target_angle - self.heading + pi) % (2 * pi) - pi
        max_turn = self.max_turn_rate / max(1.0, self.current_speed / self.turn_speed_factor)
        steering = np.clip(heading_diff, -max_turn * self.dt, max_turn * self.dt)
        self.heading += steering

    def _compute_target_speed(self):
        perf = polar_performance(self.apparent_wind_angle)
        self.target_speed = self.apparent_wind_speed * perf

    def _adjust_speed(self):
        diff = self.target_speed - self.current_speed
        self.current_speed += diff * min(1.0, self.acceleration_factor * self.dt)

    def _apply_drag(self):
        drag = self.drag_coefficient * self.current_speed * self.current_speed
        self.current_speed -= drag * self.dt 

    def _update_velocity(self):
        self.velocity = np.array([cos(self.heading), sin(self.heading)]) * self.current_speed

    def _move(self):
        self.pos += self.velocity * self.dt
        self.history.append(self.pos.copy())
        if hasattr(self, "distance_to_target") and self.distance_to_target < 0.5:
            self.path_index += 1
