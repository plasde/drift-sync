import numpy as np
from math import pi, atan2
from core.wind import WindModel
from core.sailboat import Sailboat

class SailboatEnv:
    def __init__(self, boat, wind_model, obstacles=None, goal_pos=None):
        # Link to existing components
        self.boat = Sailboat(start_pos.copy())  # Your existing Sailboat instance
        self.wind_model = WindModel()  # Your existing wind model
        
        # Environment parameters
        self.obstacles = obstacles if obstacles is not None else []
        self.goal_pos = goal_pos if goal_pos is not None else np.array([80.0, 80.0])
        self.max_steps = 1000
        
        # RL specific attributes
        self.step_count = 0
        self.done = False
        self.last_dist_to_goal = None
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset boat position (you might want to modify your Sailboat class 
        # to have a reset method rather than creating a new instance)
        self.boat.pos = self.boat.initial_pos.copy()
        self.boat.heading = self.boat.initial_heading
        self.boat.velocity = np.array([0.0, 0.0])
        self.boat.current_speed = 0.0
        
        # Reset RL state
        self.step_count = 0
        self.done = False
        self.last_dist_to_goal = np.linalg.norm(self.boat.pos - self.goal_pos)
        
        # Get initial state
        state = self._get_state_observation()
        return state
    
    def _get_state_observation(self):
        """Return the state observation for RL"""
        # Current wind vector at boat position
        wind_vec = WindModel.get_wind_at(self.boat.pos)
        
        # Normalized position (scale based on your environment size)
        norm_pos = self.boat.pos / 100.0  # Assuming environment is 100x100 units
        
        # Normalized velocity
        norm_velocity = self.boat.velocity / 20.0  # Assuming max speed is around 20 units
        
        # Heading (normalized to [0, 1])
        norm_heading = self.boat.heading / (2 * pi)
        
        # Wind relative to boat heading
        apparent_wind_vec = wind_vec - self.boat.velocity
        apparent_wind_speed = np.linalg.norm(apparent_wind_vec)
        
        if apparent_wind_speed > 0.001:
            apparent_wind_angle = atan2(apparent_wind_vec[1], apparent_wind_vec[0]) - self.boat.heading
            apparent_wind_angle = (apparent_wind_angle + pi) % (2 * pi) - pi
        else:
            apparent_wind_angle = 0
            
        norm_wind_angle = (apparent_wind_angle + pi) / (2 * pi)
        norm_wind_speed = apparent_wind_speed / 20.0  # Normalize wind speed
        
        # Vector to goal
        vec_to_goal = self.goal_pos - self.boat.pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        angle_to_goal = atan2(vec_to_goal[1], vec_to_goal[0]) - self.boat.heading
        angle_to_goal = (angle_to_goal + pi) % (2 * pi) - pi
        
        norm_dist_to_goal = np.clip(dist_to_goal / 100.0, 0, 1)  # Normalize distance
        norm_angle_to_goal = (angle_to_goal + pi) / (2 * pi)
        
        # Obstacle information (nearest 3 obstacles)
        obstacle_features = []
        
        # If obstacles exist
        if self.obstacles:
            obstacle_distances = []
            
            for obs_pos, obs_radius in self.obstacles:
                vec_to_obs = obs_pos - self.boat.pos
                dist_to_obs = np.linalg.norm(vec_to_obs) - obs_radius
                obstacle_distances.append((dist_to_obs, obs_pos, obs_radius))
                
            # Sort obstacles by distance
            obstacle_distances.sort(key=lambda x: x[0])
            
            # Get nearest 3 obstacles (or fewer if there aren't 3)
            for i in range(min(3, len(obstacle_distances))):
                dist_to_obs, obs_pos, obs_radius = obstacle_distances[i]
                vec_to_obs = obs_pos - self.boat.pos
                angle_to_obs = atan2(vec_to_obs[1], vec_to_obs[0]) - self.boat.heading
                angle_to_obs = (angle_to_obs + pi) % (2 * pi) - pi
                
                # Normalize values
                norm_dist_to_obs = np.clip(dist_to_obs / 50.0, 0, 1)
                norm_angle_to_obs = (angle_to_obs + pi) / (2 * pi)
                norm_radius = obs_radius / 10.0  # Assuming max radius is around 10 units
                
                obstacle_features.extend([norm_dist_to_obs, norm_angle_to_obs, norm_radius])
        
        # Pad with default values if fewer than 3 obstacles
        while len(obstacle_features) < 9:  # 3 obstacles * 3 features
            obstacle_features.extend([1.0, 0.5, 0.0])  # Far away, arbitrary angle, zero radius
        
        # Combine all features
        state = np.array([
            norm_pos[0], norm_pos[1],
            norm_velocity[0], norm_velocity[1],
            norm_heading,
            norm_wind_angle, norm_wind_speed,
            norm_dist_to_goal, norm_angle_to_goal,
            *obstacle_features
        ])
        
        return state
    
    def step(self, action):
        """Take one environment step with given action"""
        self.step_count += 1
        
        # Store old distance to calculate progress
        old_dist_to_goal = self.last_dist_to_goal
        
        # Get current wind
        wind_vec = self.WindModel.get_wind_at(self.boat.pos)
        
        # Apply action through boat physics
        self._apply_action(action, wind_vec)
        
        # Get new state
        new_state = self._get_state_observation()
        
        # Calculate reward
        reward, collision = self._calculate_reward(old_dist_to_goal)
        
        # Check if done
        self.done = collision or self._check_goal_reached() or self.step_count >= self.max_steps
        
        # Extra info
        info = {
            'collision': collision,
            'goal_reached': self._check_goal_reached(),
            'steps': self.step_count,
            'position': self.boat.pos,
            'speed': self.boat.current_speed
        }
        
        return new_state, reward, self.done, info
    
    def _apply_action(self, action, wind_vec):
        """Apply RL action to boat"""
        # For simplicity, action is just desired heading change
        # Scale action from [-1, 1] to realistic range
        desired_heading_change = action[0] * (self.boat.max_turn_rate * self.boat.dt)
        
        # Store original update logic to create a version that uses action input
        # instead of path following
        
        # Calculate apparent wind (as in your update method)
        apparent_wind_vec = wind_vec - self.boat.velocity
        apparent_wind_speed = np.linalg.norm(apparent_wind_vec)
        
        if apparent_wind_speed > 0.001:
            apparent_wind_angle = atan2(apparent_wind_vec[1], apparent_wind_vec[0]) - self.boat.heading
            apparent_wind_angle = (apparent_wind_angle + pi) % (2 * pi) - pi
        else:
            apparent_wind_angle = 0
            
        # Get performance from polar curve
        perf = self.boat.polar_performance(apparent_wind_angle)
        
        # Calculate target speed
        target_speed = apparent_wind_speed * perf
        
        # Apply inertia
        speed_diff = target_speed - self.boat.current_speed
        self.boat.current_speed += speed_diff * min(1.0, self.boat.acceleration_factor * self.boat.dt)
        
        # Apply drag
        drag_force = self.boat.drag_coefficient * self.boat.current_speed * self.boat.current_speed
        self.boat.current_speed -= drag_force * self.boat.dt
        
        # Apply heading change from action with speed-dependent turning rate
        max_turn_rate = self.boat.max_turn_rate / max(1.0, self.boat.current_speed / self.boat.turn_speed_factor)
        steering_angle = np.clip(desired_heading_change, -max_turn_rate * self.boat.dt, max_turn_rate * self.boat.dt)
        self.boat.heading += steering_angle
        self.boat.heading = self.boat.heading % (2 * pi)  # Keep heading in [0, 2π]
        
        # Update velocity vector
        self.boat.velocity = np.array([np.cos(self.boat.heading), np.sin(self.boat.heading)]) * self.boat.current_speed
        
        # Move boat
        self.boat.pos += self.boat.velocity * self.boat.dt
        
        # Update last distance to goal
        self.last_dist_to_goal = np.linalg.norm(self.boat.pos - self.goal_pos)
    
    def _check_collision(self):
        """Check if boat has collided with any obstacle"""
        for obs_pos, obs_radius in self.obstacles:
            distance = np.linalg.norm(self.boat.pos - obs_pos)
            if distance < obs_radius + 0.5:  # 0.5 is boat radius
                return True
        return False
    
    def _check_goal_reached(self):
        """Check if boat has reached the goal"""
        dist_to_goal = np.linalg.norm(self.boat.pos - self.goal_pos)
        return dist_to_goal < 2.0  # Goal radius
    
    def _calculate_reward(self, old_dist_to_goal):
        """Calculate reward for RL"""
        # Check if collided with obstacle
        collision = self._check_collision()
        if collision:
            return -100.0, True  # Large negative reward for collision
        
        # Check if reached goal
        if self._check_goal_reached():
            return 100.0, False  # Large positive reward for reaching goal
        
        # Progress reward (encourage moving toward goal)
        progress_reward = old_dist_to_goal - self.last_dist_to_goal
        progress_reward *= 10.0  # Scale up to make it more significant
        
        # Speed reward (encourage efficient sailing)
        speed_reward = self.boat.current_speed * 0.1
        
        # Time penalty (discourage taking too long)
        time_penalty = -0.1
        
        # Obstacle avoidance reward
        obstacle_reward = 0
        for obs_pos, obs_radius in self.obstacles:
            distance = np.linalg.norm(self.boat.pos - obs_pos) - obs_radius
            if distance < 10.0:  # Start penalizing when within 10 units of obstacle edge
                proximity_penalty = -2.0 * max(0, (10.0 - distance) / 10.0)
                obstacle_reward += proximity_penalty
        
        return progress_reward + speed_reward + time_penalty + obstacle_reward, False
