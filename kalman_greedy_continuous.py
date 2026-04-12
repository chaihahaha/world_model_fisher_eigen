import numpy as np
from typing import Tuple, Dict


class KalmanStateTracker:
    """
    Tracks state uncertainty using Kalman filter / Riccati equation.
    
    This is the KEY component from the sensor selection paper.
    """
    
    def __init__(
        self,
        state_dim: int,
        feature_dim: int = 32,
        sigma2_obs: float = 1.0,
        sigma2_proc: float = 0.01,
        init_P_scale: float = 10.0,
    ):
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.sigma2_obs = sigma2_obs
        self.sigma2_proc = sigma2_proc
        
        np.random.seed(42)
        self.W_feature = np.random.randn(feature_dim, state_dim) * 0.1
        
        # Covariance matrix P
        self.P = init_P_scale * np.eye(feature_dim)
        
        # Value parameters
        self.mu = np.zeros(feature_dim)
        
    def get_feature(self, state: np.ndarray) -> np.ndarray:
        """Extract feature from state."""
        state = np.array(state).flatten()
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        return np.tanh(state @ self.W_feature.T)
    
    def compute_value(self, state: np.ndarray) -> float:
        """Estimated value."""
        phi = self.get_feature(state)
        return np.dot(phi, self.mu)
    
    def compute_information_gain(self, state: np.ndarray) -> float:
        """
        Sensor selection criterion:
        IG(s) = φ^T P² φ / (φ^T P φ + σ²)
        
        Measures how much uncertainty would be reduced by observing this state.
        """
        phi = self.get_feature(state)
        P_phi = self.P @ phi
        numerator = np.dot(P_phi, P_phi)
        denominator = np.dot(phi, P_phi) + self.sigma2_obs
        return numerator / denominator
    
    def update(self, state: np.ndarray, reward: float):
        """Kalman update."""
        phi = self.get_feature(state)
        
        predicted = np.dot(phi, self.mu)
        innovation = reward - predicted
        
        P_phi = self.P @ phi
        kalman_gain = P_phi / (np.dot(phi, P_phi) + self.sigma2_obs)
        
        self.mu = self.mu + kalman_gain * innovation
        self.P = self.P - np.outer(kalman_gain, P_phi)
        self.P = self.P + self.sigma2_proc * np.eye(self.feature_dim)
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-8:
            self.P += (1e-8 - min_eig) * np.eye(self.feature_dim)
    
    def reset(self):
        """Reset tracker."""
        self.P = 10.0 * np.eye(self.feature_dim)
        self.mu = np.zeros(self.feature_dim)


class KalmanGreedyContinuous:
    """
    Kalman-Greedy for Continuous Actions.
    
    ALGORITHM:
    1. For action selection:
       - Sample candidate actions
       - For each action, simulate trajectory using simple physics model
       - Score = reward + β * IG
       - Select action with highest score
    2. Update uncertainty tracker with real experience
    
    NOVELTY:
    - Model-based planning with uncertainty-aware trajectory selection
    - NOT adding intrinsic rewards - using uncertainty for planning
    - Works with simple physics models (no neural networks needed)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        # Planning parameters
        n_plan_samples: int = 20,
        plan_horizon: int = 5,
        explore_weight: float = 0.7,
        gamma: float = 0.99,
        # Environment model parameters
        step_size: float = 0.1,
        env_bounds: Tuple[float, float] = (0, 10),
    ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.n_plan_samples = n_plan_samples
        self.plan_horizon = plan_horizon
        self.explore_weight = explore_weight
        self.gamma = gamma
        self.step_size = step_size
        self.env_bounds = env_bounds
        
        # Value tracker with Riccati
        self.value_tracker = KalmanStateTracker(
            state_dim=state_dim,
            feature_dim=32,
        )
        
        # Statistics
        self.steps = 0
        self.episode_rewards = []
        self.episode_reward = 0
        
    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Simple physics model: next_state = state + action * step_size
        
        This is a simple approximation - real environments may have
        more complex dynamics.
        """
        state = np.array(state).flatten()
        action = np.array(action).flatten()
        
        # Clip action
        action = np.clip(action, -self.max_action, self.max_action)
        
        # Simple movement model
        next_state = state + action * self.step_size
        
        # Apply bounds
        next_state = np.clip(
            next_state,
            self.env_bounds[0],
            self.env_bounds[1]
        )
        
        return next_state
    
    def estimate_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Estimate reward using learned value function.
        
        r(s,a) ≈ V(s') - γV(s)
        """
        v_next = self.value_tracker.compute_value(next_state)
        v_curr = self.value_tracker.compute_value(state)
        return v_next - self.gamma * v_curr
    
    def simulate_trajectory(
        self,
        start_state: np.ndarray,
        first_action: np.ndarray
    ) -> Tuple[float, float]:
        """
        Simulate one trajectory starting with first_action.
        
        Returns:
        - total_reward: estimated total reward
        - total_ig: total information gain
        """
        state = start_state
        total_reward = 0.0
        total_ig = 0.0
        
        # First step with given action
        next_state = self.predict_next_state(state, first_action)
        reward = self.estimate_reward(state, first_action, next_state)
        
        total_reward += reward
        total_ig += self.value_tracker.compute_information_gain(next_state)
        state = next_state
        
        # Continue with greedy actions
        for t in range(self.plan_horizon - 1):
            # Greedy: move toward highest value direction
            best_score = -np.inf
            best_action = np.zeros(self.action_dim)
            
            for _ in range(5):
                sample_action = np.random.uniform(
                    -self.max_action, self.max_action, self.action_dim
                )
                sample_next = self.predict_next_state(state, sample_action)
                sample_value = self.value_tracker.compute_value(sample_next)
                
                if sample_value > best_score:
                    best_score = sample_value
                    best_action = sample_action
            
            next_state = self.predict_next_state(state, best_action)
            reward = self.estimate_reward(state, best_action, next_state)
            
            discount = self.gamma ** (t + 1)
            total_reward += discount * reward
            total_ig += discount * self.value_tracker.compute_information_gain(next_state)
            state = next_state
        
        return total_reward, total_ig
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action via model-based planning.
        
        This is the KEY NOVEL COMPONENT - planning with uncertainty-aware scoring.
        """
        state = np.array(state).flatten()
        
        # Sample candidate actions
        best_score = -np.inf
        best_action = np.zeros(self.action_dim)
        
        for _ in range(self.n_plan_samples):
            action = np.random.uniform(
                -self.max_action, self.max_action, self.action_dim
            )
            
            # Simulate trajectory
            total_reward, total_ig = self.simulate_trajectory(state, action)
            
            # Combined score: reward + explore_weight * IG
            score = total_reward + self.explore_weight * total_ig
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience and update models."""
        # Update value tracker
        self.value_tracker.update(state, reward)
        
        # Track episode reward
        self.episode_reward += reward
        self.steps += 1
        
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        eig = np.linalg.eigvalsh(self.value_tracker.P)
        return {
            'episodes': len(self.episode_rewards),
            'recent_reward': float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0.0,
            'P_trace': float(np.trace(self.value_tracker.P)),
            'P_max_eig': float(np.max(eig)),
            'steps': self.steps,
        }

