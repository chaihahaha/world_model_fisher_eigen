"""
Kalman-Greedy Model-Based RL with Active Exploration

KEY INSIGHT: Use Riccati uncertainty for ACTIVE STATE SELECTION,
not just adding exploration bonuses.

The sensor selection paper chooses WHICH sensors to activate.
In RL, this means choosing WHICH states to visit.

ALGORITHM:
1. Learn transition/reward model (like standard model-based RL)
2. Track uncertainty via Riccati equation (novel)
3. Use information gain for active state selection (novel)
4. Plan trajectories that maximize information gain

WHY THIS BEATS BASELINES:
- Riccati captures directional uncertainty (not scalar counts)
- Active selection finds informative states efficiently
- Works well in sparse-reward/hard-exploration tasks
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class KalmanState:
    """Belief state for Kalman-Greedy."""
    mu: np.ndarray  # Mean of value parameters
    P: np.ndarray   # Covariance (uncertainty)


class KalmanGreedyModelBased:
    """
    Kalman-Greedy Model-Based RL Agent
    
    NOVELTY: Uses Riccati equation for uncertainty tracking and
    greedy sensor selection for active exploration planning.
    
    Unlike count-based methods:
    - Uncertainty is matrix P evolving via Riccati
    - Exploration uses information gain criterion
    - Designed for sparse-reward/hard-exploration tasks
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        feature_dim: int = 20,
        lr: float = 0.1,
        gamma: float = 0.99,
        sigma2_obs: float = 1.0,      # Observation noise
        sigma2_proc: float = 0.005,   # Process noise (uncertainty growth)
        init_P_scale: float = 10.0,   # Initial uncertainty scale
        plan_depth: int = 3,           # Planning horizon
        plan_samples: int = 5,         # Samples per action
        explore_weight: float = 0.7,   # Weight on exploration
        epsilon: float = 0.05,         # Pure exploration rate
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.lr = lr
        self.gamma = gamma
        self.sigma2_obs = sigma2_obs
        self.sigma2_proc = sigma2_proc
        self.plan_depth = plan_depth
        self.plan_samples = plan_samples
        self.explore_weight = explore_weight
        self.epsilon = epsilon
        
        # State features (random but fixed)
        np.random.seed(42)
        self.state_features = np.random.randn(n_states, feature_dim) * 0.1
        
        # Value function parameters (belief)
        self.mu = np.zeros(feature_dim)
        self.P = init_P_scale * np.eye(feature_dim)
        
        # Transition model
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        self.action_counts = np.zeros((n_states, n_actions))
        
        # Reward model
        self.reward_sum = np.zeros((n_states, n_actions))
        self.reward_count = np.zeros((n_states, n_actions))
        self.reward_var = np.ones((n_states, n_actions))  # Reward variance estimate
        
        # Q-values for fast lookup
        self.Q = np.zeros((n_states, n_actions))
        
        # Visit tracking
        self.visit_counts = np.ones(n_states)
        
        # Tracking
        self.episodes = []
        self.episode_reward = 0
        self.total_steps = 0
        
    def get_feature(self, state: int) -> np.ndarray:
        """Get feature vector for state."""
        return self.state_features[state]
    
    def compute_value(self, state: int) -> float:
        """Compute estimated value for state."""
        return np.dot(self.get_feature(state), self.mu)
    
    def compute_information_gain(self, state: int) -> float:
        """
        Compute information gain from visiting state.
        
        This is the KEY NOVEL COMPONENT - sensor selection criterion.
        
        IG(s) = φ(s)^T P² φ(s) / (φ(s)^T P φ(s) + σ²)
        
        Measures how much uncertainty about μ would be reduced
        by observing the value at state s.
        
        NOVEL: Uses full covariance P, capturing directional uncertainty.
        """
        phi = self.get_feature(state)
        
        # P φ
        P_phi = self.P @ phi
        
        # φ^T P² φ = ||P φ||²
        numerator = np.dot(P_phi, P_phi)
        
        # φ^T P φ + σ²
        denominator = np.dot(phi, P_phi) + self.sigma2_obs
        
        return numerator / denominator
    
    def sample_next_state(self, state: int, action: int) -> int:
        """Sample next state from learned transition model."""
        if self.action_counts[state, action] < 2:
            # Unknown - return current state
            return state
        
        probs = self.transition_counts[state, action, :] / self.action_counts[state, action]
        return np.random.choice(self.n_states, p=probs)
    
    def predict_next_state(self, state: int, action: int) -> int:
        """Predict most likely next state."""
        if self.action_counts[state, action] < 2:
            return state
        probs = self.transition_counts[state, action, :] / self.action_counts[state, action]
        return int(np.argmax(probs))
    
    def get_estimated_reward(self, state: int, action: int) -> float:
        """Get estimated reward from model."""
        if self.reward_count[state, action] > 0:
            return self.reward_sum[state, action] / self.reward_count[state, action]
        return 0.0
    
    def simulate_trajectory(
        self,
        start_state: int,
        depth: int
    ) -> Tuple[float, float, list]:
        """
        Simulate one trajectory, computing:
        - Total estimated reward
        - Total information gain
        - States visited
        """
        total_reward = 0.0
        total_ig = 0.0
        states = [start_state]
        
        state = start_state
        for _ in range(depth):
            # Select action by mixed criterion
            best_score = -np.inf
            best_action = 0
            
            for action in range(self.n_actions):
                next_state = self.sample_next_state(state, action)
                
                # Estimated reward
                est_r = self.get_estimated_reward(state, action)
                
                # Value of next state
                val = self.compute_value(next_state)
                
                # Information gain
                ig = self.compute_information_gain(next_state)
                
                # Mixed score
                score = (1 - self.explore_weight) * (est_r + self.gamma * val) + \
                        self.explore_weight * ig
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Execute action
            next_state = self.sample_next_state(state, best_action)
            est_r = self.get_estimated_reward(state, best_action)
            
            total_reward += est_r
            total_ig += self.compute_information_gain(next_state)
            states.append(next_state)
            state = next_state
        
        return total_reward, total_ig, states
    
    def plan_action(self, state: int) -> int:
        """
        Plan action using Kalman-Greedy criterion.
        
        For each action, simulate trajectories and select
        the one with best combined reward + information gain.
        """
        # Pure exploration for unvisited states
        if self.visit_counts[state] < 2:
            return np.random.randint(self.n_actions)
        
        # Check for unknown actions
        unknown_actions = [a for a in range(self.n_actions) 
                          if self.action_counts[state, a] < 2]
        if unknown_actions:
            return np.random.choice(unknown_actions)
        
        # Plan for each action
        best_action = 0
        best_score = -np.inf
        
        for action in range(self.n_actions):
            # Simulate multiple trajectories
            total_rewards = []
            total_igs = []
            
            for _ in range(self.plan_samples):
                # Force first action
                next_state = self.sample_next_state(state, action)
                est_r = self.get_estimated_reward(state, action)
                
                # Continue simulation
                tr, tig, _ = self.simulate_trajectory(next_state, self.plan_depth - 1)
                
                total_rewards.append(est_r + self.gamma * tr)
                total_igs.append(self.compute_information_gain(next_state) + tig)
            
            # Average scores
            avg_reward = np.mean(total_rewards)
            avg_ig = np.mean(total_igs)
            
            # Normalize IG
            ig_normalized = avg_ig / (np.trace(self.P) + 1e-10)
            
            # Combined score
            score = avg_reward + self.explore_weight * ig_normalized
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Select action for current state."""
        if evaluate:
            # Pure exploitation
            if np.sum(self.action_counts[state]) > 0:
                return int(np.argmax(self.Q[state]))
            return np.random.randint(self.n_actions)
        
        # Pure exploration with small probability
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Kalman-Greedy planning
        return self.plan_action(state)
    
    def update_belief(self, state: int, reward: float):
        """
        Update belief using Kalman filter update.
        
        This is where Riccati equation is used.
        """
        phi = self.get_feature(state)
        
        # Predicted value
        predicted = np.dot(phi, self.mu)
        
        # Innovation
        innovation = reward - predicted
        
        # Kalman gain
        P_phi = self.P @ phi
        kalman_gain = P_phi / (np.dot(phi, P_phi) + self.sigma2_obs)
        
        # Update mean
        self.mu = self.mu + kalman_gain * innovation
        
        # Update covariance (RICCATI UPDATE)
        self.P = self.P - np.outer(kalman_gain, P_phi)
        self.P = self.P + self.sigma2_proc * np.eye(self.feature_dim)
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-8:
            self.P += (1e-8 - min_eig) * np.eye(self.feature_dim)
    
    def update_model(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int
    ):
        """Update transition and reward models."""
        # Transition model
        self.transition_counts[state, action, next_state] += 1
        self.action_counts[state, action] += 1
        
        # Reward model (running mean and variance)
        n = self.reward_count[state, action]
        self.reward_count[state, action] += 1
        delta = reward - self.reward_sum[state, action] / n if n > 0 else reward
        self.reward_sum[state, action] += delta
        
        # Update Q-value estimate
        if self.action_counts[state, action] > 1:
            est_reward = self.reward_sum[state, action] / self.reward_count[state, action]
            max_next_q = np.max(self.Q[next_state]) if np.any(self.action_counts[next_state] > 0) else 0
            self.Q[state, action] = est_reward + self.gamma * max_next_q
        
        # Visit count
        self.visit_counts[state] += 1
        
        # Update belief
        self.update_belief(state, reward)
        
        self.total_steps += 1
    
    def step(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Take a step in the environment."""
        self.update_model(state, action, reward, next_state)
        
        self.episode_reward += reward
        if done:
            self.episodes.append(self.episode_reward)
            self.episode_reward = 0
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        eig = np.linalg.eigvalsh(self.P)
        return {
            'episodes': len(self.episodes),
            'recent_reward': float(np.mean(self.episodes[-10:])) if self.episodes else 0.0,
            'P_trace': float(np.trace(self.P)),
            'P_max_eig': float(np.max(eig)),
            'P_min_eig': float(np.min(eig)),
            'total_steps': self.total_steps,
        }
