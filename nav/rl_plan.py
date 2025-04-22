import numpy as np

def rl_plan(start, goal, wind_field, config=None):
    """
    Plans a path using a reinforcement learning agent.

    Parameters:
    - start: np.array([x, y])
    - goal: np.array([x, y])
    - wind_field: function or grid with wind vectors
    - config: dict with optional parameters

    Returns:
    - List of np.array waypoints
    """
    # Placeholder agent — will be replaced with learned policy
    agent = DummyRLAgent()
    return agent.plan(start, goal)


class DummyRLAgent:
    def plan(self, start, goal):
        # Straight-line path with 10 steps
        steps = 10
        path = [start + (goal - start) * i / steps for i in range(1, steps + 1)]
        return path
