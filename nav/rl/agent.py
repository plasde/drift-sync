import numpy as np
import random

class SimpleRLAgent:
    """A very simple RL agent using epsilon-greedy Q-learning"""
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Simple linear Q-function approximation
        self.weights = np.random.randn(state_size, action_size) * 0.1
        
    def get_action(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            # Exploration: random action
            return np.array([np.random.uniform(-1, 1)])
        else:
            # Exploitation: use policy
            q_values = np.dot(state, self.weights)
            # For continuous action space, we'll use a simple approach
            # Map the highest Q-value to an action in [-1, 1]
            best_action_idx = np.argmax(q_values)
            return np.array([(best_action_idx / (self.action_size - 1)) * 2 - 1])
    
    def learn(self, state, action, reward, next_state, done):
        # Convert continuous action to discretized index
        action_idx = int((action[0] + 1) / 2 * (self.action_size - 1))
        action_idx = np.clip(action_idx, 0, self.action_size - 1)
        
        # Current Q value
        current_q = np.dot(state, self.weights[:, action_idx])
        
        # Next Q value (max over all actions)
        next_q = np.max(np.dot(next_state, self.weights)) if not done else 0
        
        # Target Q value
        target_q = reward + self.discount_factor * next_q
        
        # Update weights
        delta = target_q - current_q
        self.weights[:, action_idx] += self.learning_rate * delta * state
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay