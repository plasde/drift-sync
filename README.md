## Autonomous sailing through reinforcement learning

This project investigates autonomous sailing through the integration of path planning, reinforcement learning, and environmental data. The system is divided into hierarchical modules for global routing, local navigation, and adaptive learning.

### Global Path Planning

Objective: Compute long-distance routes under environmental constraints.

Method: A* search algorithm with sailing time as the cost function. Cost is adjusted using wind and meteorological data (GRIB).


### Local Path Planning

Objective: Navigate between waypoints while avoiding obstacles.

Method: Later

Deployment: Later


### Adaptive Learning

Objective: Improve navigation strategies through experience.

Method: Deep Reinforcement Learning (actor–critic approaches).


### System Integration

Architecture: Hierarchical control where the global planner provides waypoints, the local planner executes short-term navigation, and the adaptive module refines decisions.
