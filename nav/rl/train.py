import numpy as np
import time
from nav.rl.env import SailboatEnv
from nav.rl.agent import SimpleRLAgent

def train_rl_agent(boat, wind_model, obstacles, goal_pos, episodes=1000):
    # Create environment
    env = SailboatEnv(boat, wind_model, obstacles, goal_pos)
    
    # Create agent
    state_size = 18  # Matches our state representation
    action_size = 11  # Discretized actions from -1 to 1
    agent = SimpleRLAgent(state_size, action_size)
    
    # Training loop
    best_reward = -float('inf')
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get action
            action = agent.get_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and stats
            state = next_state
            total_reward += reward
            steps += 1
        
        # Print episode stats
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Steps: {steps}")
        print(f"  Goal reached: {info['goal_reached']}, Collision: {info['collision']}")
        
        # Save best agent
        if total_reward > best_reward:
            best_reward = total_reward
            # Here you could save the agent's weights
    
    return agent