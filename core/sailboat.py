import numpy as np
from math import atan2, pi, sin, cos, radians
from core.polar import polar_performance
from data.environment_data import WIND_SPEED, DT, DRAG_COEFFICIENT, ACCELERATION_FACTOR, MAX_TURN_RATE, TURN_SPEED_FACTOR

# Boat class
class Sailboat:
    def __init__(self, pos, heading):
        self.pos = pos
        self.heading = heading
        self.history = [pos.copy()]
        self.current_speed = 0
        self.velocity = np.array([0.0, 0.0])
        self.path = []
        self.path_index = 0
    
    def set_path(self, path):
        self.path = path
        self.path_index = 0

    def update(self, wind_vec):
        if self.path_index < len(self.path):
            target = self.path[self.path_index]
        
            # Calculate apparent wind (vector subtraction: apparent_wind = true_wind - boat_velocity)
            apparent_wind_vec = wind_vec - self.velocity
            apparent_wind_speed = np.linalg.norm(apparent_wind_vec)
        
            # Calculate apparent wind angle relative to boat heading
            if apparent_wind_speed > 0.001:  # Avoid division by zero
                apparent_wind_angle = atan2(apparent_wind_vec[1], apparent_wind_vec[0]) - self.heading
                apparent_wind_angle = (apparent_wind_angle + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
            else:
                apparent_wind_angle = 0
            
            # Get performance from polar curve using apparent wind
            perf = polar_performance(apparent_wind_angle)
        
            # Calculate target speed based on polar performance
            target_speed = apparent_wind_speed * perf
        
            # Apply inertia - boat doesn't change speed instantly
            speed_diff = target_speed - self.current_speed
            self.current_speed += speed_diff * min(1.0, ACCELERATION_FACTOR * DT)
        
            # Calculate drag as a function of speed squared (more realistic)
            drag_force = DRAG_COEFFICIENT * self.current_speed * self.current_speed
            self.current_speed -= drag_force * DT
        
            # Calculate vector to next waypoint
            direction_to_target = target - self.pos
            distance_to_target = np.linalg.norm(direction_to_target)
            target_angle = atan2(direction_to_target[1], direction_to_target[0])
        
            # Calculate desired heading change
            heading_diff = (target_angle - self.heading + pi) % (2 * pi) - pi
        
            # Implement more realistic steering - slower at higher speeds
            max_turn_rate = MAX_TURN_RATE / max(1.0, self.current_speed / TURN_SPEED_FACTOR)
            steering_angle = np.clip(heading_diff, -max_turn_rate * DT, max_turn_rate * DT)
            self.heading += steering_angle
        
            # Update velocity vector based on new heading and speed
            self.velocity = np.array([cos(self.heading), sin(self.heading)]) * self.current_speed
        
            # Move boat
            self.pos += self.velocity * DT
        
            # Check if reached waypoint (use adaptive radius based on speed)
            arrival_radius = max(0.1, min(0.3, self.current_speed * 0.5))
            if distance_to_target < arrival_radius:
                self.path_index += 1

        # Store path
        self.history.append(self.pos.copy())