import numpy as np

class BasicControllerAgent:
    def __init__(self, gain=1.5, max_rudder=0.2):
        self.gain = gain
        self.max_rudder = max_rudder

    def get_action(self, observation):
        """
        observation = [heading, rel_angle_to_goal, apparent_wind_angle, speed]
        """
        rel_angle = observation[1]
        rudder = self.gain * rel_angle
        rudder = np.clip(rudder, -self.max_rudder, self.max_rudder)
        return [rudder]
