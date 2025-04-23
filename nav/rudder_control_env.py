import numpy as np
from core.sailboat import Sailboat

class RudderControlEnv:
    def __init__(self, start_pos, goal_pos, wind_fn, dt=0.1):
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.wind_fn = wind_fn  # Function that returns wind vector given pos or time
        self.dt = dt
        self.max_steps = 500
        self.goal_threshold = 2.0
        self.reset()

    def reset(self):
        self.boat = Sailboat(pos=self.start_pos.copy(), heading=0.0, boat_type="boat1", dt=self.dt)
        self.steps = 0
        wind = self.wind_fn(self.boat.pos)
        self.boat._compute_apparent_wind(wind)
        self.prev_distance = self._distance_to_goal()
        return self._get_observation()

    def step(self, action):
        rudder_delta = action[0]  # Single float: change in heading
        self.boat.heading += np.clip(rudder_delta, -0.1, 0.1)  # Limit turn rate manually here

        wind = self.wind_fn(self.boat.pos)
        self.boat._compute_apparent_wind(wind)  # use just the dynamics
        self.boat._compute_target_speed()
        self.boat._adjust_speed()
        self.boat._apply_drag()
        self.boat._update_velocity()
        self.boat._move()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        print(f"Pos: {self.boat.pos}, Speed: {self.boat.current_speed:.2f}, Heading: {np.degrees(self.boat.heading):.1f}, Reward: {reward:.2f}, Done: {done}")

        self.steps += 1
        return obs, reward, done, info

    def _get_observation(self):
        vec_to_goal = self.goal_pos - self.boat.pos
        angle_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        rel_angle = (angle_to_goal - self.boat.heading + np.pi) % (2 * np.pi) - np.pi
        return np.array([
            self.boat.heading,
            rel_angle,
            self.boat.apparent_wind_angle,
            self.boat.current_speed
        ], dtype=np.float32)
    
    def _distance_to_goal(self):
        return np.linalg.norm(self.goal_pos - self.boat.pos)

    def _compute_reward(self):
        curr_distance = self._distance_to_goal()
        progress = self.prev_distance - curr_distance
        self.prev_distance = curr_distance

        step_penalty = -0.1
        reached_goal = curr_distance < self.goal_threshold
        goal_bonus = 100.0 if reached_goal else 0.0

        return 25.0 * progress + step_penalty + goal_bonus

    def _check_done(self):
        dist_to_goal = np.linalg.norm(self.goal_pos - self.boat.pos)
        return dist_to_goal < self.goal_threshold or self.steps >= self.max_steps

    def render(self):
        print(f"Step {self.steps}: Pos {self.boat.pos}, Heading {self.boat.heading:.2f}")
