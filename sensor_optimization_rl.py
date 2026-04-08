"""
SENSOR OPTIMIZATION FOR RL EXPLORATION

Adapting sensor placement optimization from Kalman filtering to RL exploration.

Key insight from sensor optimization:
- In Kalman filtering, we select sensors to minimize state estimation uncertainty
- Uncertainty is measured by the error covariance matrix P
- Optimal sensor selection maximizes information gain (Fisher Information Matrix)

Adaptation to RL:
- Treat states as "sensors" that provide information about the environment
- Track uncertainty (covariance) of value estimates for each state
- Explore states that maximize information gain (minimize uncertainty reduction)

Optimality criteria:
- D-optimality: Maximize det(I) where I is Fisher Information Matrix
  Equivalent to minimizing volume of uncertainty ellipsoid
- A-optimality: Minimize trace(P) where P is covariance matrix
  Minimizes average variance across dimensions
- E-optimality: Minimize λ_max(P)
  Minimizes worst-case uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import random
import warnings
import time
from typing import Dict, Tuple, Optional, List
import json
import os

warnings.filterwarnings('ignore')

try:
    import minigrid  # Must import before gymnasium to register environments
    import gymnasium as gym
except ImportError:
    import subprocess
    subprocess.call(["pip", "install", "minigrid", "gymnasium", "-q"])
    import minigrid  # Must import before gymnasium to register environments
    import gymnasium as gym


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_state(obs: dict) -> np.ndarray:
    """Extract state vector from observation."""
    if isinstance(obs, dict):
        return obs['image'].flatten()
    return np.array(obs).flatten()


def state_to_hashable(state: np.ndarray) -> tuple:
    """Convert state to hashable form for dictionary keys."""
    return tuple(state.astype(np.int8))


# ============================================================================
# SENSOR OPTIMIZATION MODULE
# ============================================================================

class SensorOptimizationEngine:
    """
    Implements sensor optimization criteria for exploration bonus calculation.
    
    Based on sensor placement optimization for Kalman filtering:
    - Track Fisher Information Matrix (or covariance) for each state
    - Compute exploration bonus based on uncertainty measures
    
    Key matrices:
    - P: Covariance matrix (uncertainty) - initialized to high values
    - I: Fisher Information Matrix - accumulated from observations
    
    Relationship: P = I^(-1) (under certain conditions)
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 init_uncertainty: float = 10.0,
                 decay_rate: float = 0.99,
                 min_uncertainty: float = 0.01):
        """
        Args:
            state_dim: Dimensionality of state space
            action_dim: Number of actions
            init_uncertainty: Initial uncertainty (diagonal of P)
            decay_rate: How fast uncertainty decreases per visit
            min_uncertainty: Minimum uncertainty floor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.init_uncertainty = init_uncertainty
        self.decay_rate = decay_rate
        self.min_uncertainty = min_uncertainty
        
        # Track visit counts per state
        self.state_counts: Dict[tuple, int] = defaultdict(int)
        
        # Track uncertainty (simplified: scalar per state, not full matrix)
        # In full implementation, this would be a covariance matrix
        self.state_uncertainty: Dict[tuple, np.ndarray] = {}
        
        # Fisher Information accumulator (simplified)
        self.fisher_info: Dict[tuple, float] = defaultdict(float)
        
        # For D-optimality: track determinant of information matrix
        self.info_determinant: Dict[tuple, float] = defaultdict(lambda: 1.0)
        
        # For A-optimality: track trace of covariance
        self.cov_trace: Dict[tuple, float] = defaultdict(lambda: init_uncertainty * state_dim)
    
    def get_state_key(self, state: np.ndarray) -> tuple:
        """Get hashable key for state."""
        return state_to_hashable(state)
    
    def update_state(self, state: np.ndarray, action: int):
        """Update uncertainty estimates after visiting state with action."""
        state_key = self.get_state_key(state)
        
        # Update visit count
        self.state_counts[state_key] += 1
        count = self.state_counts[state_key]
        
        # Update Fisher Information (accumulates with visits)
        # More visits = more information = higher Fisher Info
        self.fisher_info[state_key] += 1.0 / count
        
        # Update determinant (for D-optimality)
        # Determinant grows with information accumulation
        self.info_determinant[state_key] *= (1.0 + 0.1 / count)
        
        # Update trace (for A-optimality) - decreases with visits
        current_trace = self.cov_trace[state_key]
        self.cov_trace[state_key] = max(
            self.min_uncertainty * self.state_dim,
            current_trace * self.decay_rate
        )
    
    def get_d_optimality_bonus(self, state: np.ndarray) -> float:
        """
        D-optimality exploration bonus.
        
        D-optimality maximizes det(I) where I is Fisher Information Matrix.
        High determinant = high information = low uncertainty.
        
        Exploration bonus: inverse of determinant (explore low-information states)
        
        Mathematically:
            det(I) = product of eigenvalues of I
            Volume of uncertainty ellipsoid ∝ 1/sqrt(det(I))
            
        So we want to explore states with LOW det(I), i.e., HIGH uncertainty.
        """
        state_key = self.get_state_key(state)
        det_I = self.info_determinant[state_key]
        
        # Bonus is higher for lower determinant (more uncertain)
        # Use log for numerical stability
        bonus = 1.0 / (np.log1p(det_I) + 1.0)
        return bonus
    
    def get_a_optimality_bonus(self, state: np.ndarray) -> float:
        """
        A-optimality exploration bonus.
        
        A-optimality minimizes trace(P) where P is covariance matrix.
        Trace = sum of variances = average uncertainty.
        
        Exploration bonus: proportional to trace (explore high-uncertainty states)
        
        Mathematically:
            trace(P) = Σᵢ Pᵢᵢ = Σᵢ Var(xᵢ)
            
        We want to explore states with HIGH trace(P), i.e., HIGH average variance.
        """
        state_key = self.get_state_key(state)
        trace_P = self.cov_trace[state_key]
        
        # Bonus is higher for higher trace (more uncertain)
        bonus = trace_P / self.init_uncertainty
        return bonus
    
    def get_e_optimality_bonus(self, state: np.ndarray) -> float:
        """
        E-optimality exploration bonus.
        
        E-optimality minimizes λ_max(P) - the maximum eigenvalue.
        This minimizes worst-case uncertainty.
        
        Exploration bonus: proportional to estimated max eigenvalue.
        
        Mathematically:
            λ_max(P) ≤ trace(P) (by properties of eigenvalues)
            λ_max(P) ≥ trace(P) / n (average eigenvalue)
            
        We approximate λ_max using trace and count-based estimates.
        """
        state_key = self.get_state_key(state)
        count = self.state_counts[state_key]
        
        # Estimate max eigenvalue from trace and visit count
        # With few visits, uncertainty is concentrated (high max eigenvalue)
        # With many visits, uncertainty spreads out (lower max eigenvalue)
        trace_P = self.cov_trace[state_key]
        
        # Approximation: λ_max ≈ trace / effective_dimension
        # effective_dimension increases with exploration
        effective_dim = min(self.state_dim, 1.0 + 0.1 * count)
        lambda_max_est = trace_P / effective_dim
        
        bonus = lambda_max_est / self.init_uncertainty
        return bonus
    
    def get_combined_bonus(self, state: np.ndarray, 
                          weights: Dict[str, float] = None) -> float:
        """
        Combined exploration bonus using multiple optimality criteria.
        
        Args:
            state: Current state
            weights: Weights for D, A, E optimality (default: equal)
        """
        if weights is None:
            weights = {'D': 1.0, 'A': 1.0, 'E': 1.0}
        
        d_bonus = self.get_d_optimality_bonus(state)
        a_bonus = self.get_a_optimality_bonus(state)
        e_bonus = self.get_e_optimality_bonus(state)
        
        # Normalize each bonus
        d_bonus /= (weights['D'] + 1.0)
        a_bonus /= (weights['A'] + 1.0)
        e_bonus /= (weights['E'] + 1.0)
        
        combined = (
            weights['D'] * d_bonus +
            weights['A'] * a_bonus +
            weights['E'] * e_bonus
        )
        
        return combined
    
    def get_state_uncertainty_estimate(self, state: np.ndarray) -> float:
        """Get overall uncertainty estimate for a state."""
        state_key = self.get_state_key(state)
        count = self.state_counts[state_key]
        
        # Count-based uncertainty (like count-based exploration)
        count_uncertainty = 1.0 / np.sqrt(max(count, 1))
        
        # Covariance-based uncertainty
        cov_uncertainty = self.cov_trace[state_key] / self.init_uncertainty
        
        return 0.5 * count_uncertainty + 0.5 * cov_uncertainty


# ============================================================================
# Q-NETWORK WITH UNCERTAINTY ESTIMATION
# ============================================================================

class UncertaintyQNetwork(nn.Module):
    """
    Q-network with uncertainty estimation.
    
    Uses ensemble or dropout-based uncertainty estimation.
    For simplicity, we use a single network with state-dependent
    uncertainty based on visitation counts.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values for state."""
        return self.forward(state)


# ============================================================================
# SENSOR OPTIMIZATION RL AGENT
# ============================================================================

class SensorOptimizationAgent:
    """
    RL Agent using sensor optimization for exploration.
    
    Combines:
    1. Q-learning for value estimation
    2. Sensor optimization criteria for exploration bonus
    3. Experience replay for sample efficiency
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 optimality: str = 'combined',
                 exploration_weight: float = 1.0,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 replay_size: int = 10000,
                 hidden_dim: int = 128):
        """
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            optimality: 'D', 'A', 'E', or 'combined'
            exploration_weight: Weight of exploration bonus
            learning_rate: Learning rate for Q-network
            gamma: Discount factor
            batch_size: Batch size for training
            replay_size: Experience replay buffer size
            hidden_dim: Hidden layer dimension
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimality = optimality
        self.exploration_weight = exploration_weight
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Q-network
        self.q_network = UncertaintyQNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate,
            eps=1e-5
        )
        
        # Experience replay
        self.memory = deque(maxlen=replay_size)
        
        # Sensor optimization engine
        self.sensor_engine = SensorOptimizationEngine(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Statistics tracking
        self.stats = {
            'visits': defaultdict(int),
            'bonus_history': [],
            'q_values_history': []
        }
    
    def select_action(self, state: np.ndarray, 
                     epsilon: float = 0.0,
                     temperature: float = 1.0) -> int:
        """
        Select action using Q-values + exploration bonus.
        
        Action selection: argmax_a [Q(s,a) + β * bonus(s,a)]
        
        Args:
            state: Current state
            epsilon: Epsilon-greedy exploration rate
            temperature: Softmax temperature
        """
        state_key = self.sensor_engine.get_state_key(state)
        self.stats['visits'][state_key] += 1
        
        # Random exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.get_q_values(state_tensor).numpy()[0]
        
        # Get exploration bonus (state-dependent, not action-dependent)
        if self.optimality == 'D':
            bonus = self.sensor_engine.get_d_optimality_bonus(state)
        elif self.optimality == 'A':
            bonus = self.sensor_engine.get_a_optimality_bonus(state)
        elif self.optimality == 'E':
            bonus = self.sensor_engine.get_e_optimality_bonus(state)
        else:  # combined
            bonus = self.sensor_engine.get_combined_bonus(state)
        
        # Track bonus
        self.stats['bonus_history'].append(bonus)
        
        # Add bonus uniformly to all actions (state-level exploration)
        q_with_bonus = q_values + self.exploration_weight * bonus
        
        # Softmax selection with temperature
        if temperature > 0:
            # Numerically stable softmax
            q_shifted = q_with_bonus - np.max(q_with_bonus)
            exp_q = np.exp(q_shifted / temperature)
            probs = exp_q / exp_q.sum()
            
            # Handle NaN/inf
            probs = np.nan_to_num(probs, nan=1.0/self.action_dim)
            probs = probs / probs.sum()
            
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = np.argmax(q_with_bonus)
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, 
                        done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
        # Update sensor optimization engine
        self.sensor_engine.update_state(state, action)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_values.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            'unique_states_visited': len(self.stats['visits']),
            'total_visits': sum(self.stats['visits'].values()),
            'avg_bonus': np.mean(self.stats['bonus_history'][-100:]) if self.stats['bonus_history'] else 0,
            'max_bonus': max(self.stats['bonus_history']) if self.stats['bonus_history'] else 0
        }


# ============================================================================
# BASELINE AGENTS
# ============================================================================

class CountBasedAgent:
    """Count-based exploration baseline (like MBIE-EB or similar)."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 exploration_weight: float = 1.0,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 batch_size: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exploration_weight = exploration_weight
        
        self.q_network = UncertaintyQNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.batch_size = batch_size
        
        # State visit counts
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(lambda: np.zeros(action_dim))
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0, temperature: float = 1.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_key = tuple(state.astype(np.int8))
        self.state_counts[state_key] += 1
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.get_q_values(state_tensor).numpy()[0]
        
        # Count-based bonus: N(s,a)^(-1/2)
        counts = self.state_action_counts[state_key]
        bonus = self.exploration_weight / np.sqrt(counts + 1)
        
        q_with_bonus = q_values + bonus
        
        # Handle potential numerical issues
        q_with_bonus = np.nan_to_num(q_with_bonus, nan=0.0)
        action = np.argmax(q_with_bonus)
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool):
        state_key = tuple(state.astype(np.int8))
        self.state_action_counts[state_key][action] += 1
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(q_values.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class RandomAgent:
    """Pure random exploration baseline."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 batch_size: int = 32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_network = UncertaintyQNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.batch_size = batch_size
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1, temperature: float = 1.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.get_q_values(state_tensor).numpy()[0]
        
        return np.argmax(q_values)
    
    def store_experience(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(q_values.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def create_environment(env_name: str = "MiniGrid-Empty-Random-5x5-v0"):
    """Create and return environment."""
    return gym.make(env_name)


def run_episode(env, agent, max_steps: int = 100, 
                epsilon: float = 0.0,
                temperature: float = 1.0,
                train_steps: int = 1,
                use_distance_reward: bool = True) -> Tuple[float, dict]:
    """
    Run one episode.
    
    Returns:
        total_reward: Total reward obtained
        stats: Episode statistics
    """
    obs, _ = env.reset()
    state = extract_state(obs)
    total_reward = 0
    steps = 0
    
    episode_stats = {
        'rewards': [],
        'bonuses': [],
        'states_visited': set()
    }
    
    # Store initial position for distance reward
    initial_pos = None
    if use_distance_reward and hasattr(env.unwrapped, 'goal_pos'):
        initial_pos = env.unwrapped.goal_pos.copy()
    
    for step in range(max_steps):
        action = agent.select_action(state, epsilon=epsilon, temperature=temperature)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(next_obs)
        done = terminated or truncated
        
        # Add distance-based reward for Empty environment
        if use_distance_reward and reward == 0 and hasattr(env.unwrapped, 'goal_pos'):
            agent_pos = (env.unwrapped.horizontal_pos, env.unwrapped.vertical_pos)
            goal_pos = env.unwrapped.goal_pos
            dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            reward = -1.0 / (dist + 1)  # Closer = less negative
        
        agent.store_experience(state, action, reward, next_state, done)
        
        total_reward += reward
        episode_stats['rewards'].append(reward)
        episode_stats['states_visited'].add(tuple(next_state.astype(np.int8)))
        
        state = next_state
        steps += 1
        
        if done:
            break
    
    # Training steps
    for _ in range(train_steps):
        agent.train_step()
    
    episode_stats['steps'] = steps
    
    return total_reward, episode_stats


def train_agent(env, agent, n_episodes: int, max_steps: int = 100,
                epsilon_start: float = 0.3, epsilon_end: float = 0.05,
                temperature: float = 1.0,
                train_steps: int = 1,
                verbose: bool = True,
                use_distance_reward: bool = True) -> List[float]:
    """
    Train agent for n episodes.
    
    Returns:
        rewards: List of episode rewards
    """
    rewards = []
    
    for ep in range(n_episodes):
        # Decay epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - ep / n_episodes)
        
        # Run episode
        episode_reward, _ = run_episode(env, agent, max_steps, epsilon, temperature, train_steps, use_distance_reward)
        rewards.append(episode_reward)
        
        if verbose and (ep + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {ep+1}/{n_episodes}, Avg Reward (last 10): {avg_reward:.3f}, Epsilon: {epsilon:.3f}")
    
    return rewards


def run_comparison_experiment(env_name: str = "MiniGrid-Empty-8x8-v0",
                             n_seeds: int = 5,
                             n_episodes: int = 100,
                             max_steps: int = 100,
                             exploration_weight: float = 1.0,
                             use_distance_reward: bool = True):
    """
    Run comparison experiment between sensor optimization and baselines.
    
    Args:
        env_name: Gym environment name
        n_seeds: Number of random seeds
        n_episodes: Episodes per seed
        max_steps: Max steps per episode
        exploration_weight: Exploration bonus weight
    """
    print("="*80)
    print("SENSOR OPTIMIZATION FOR RL EXPLORATION")
    print("Comparison with Count-Based and Random Baselines")
    print("="*80)
    
    # Results storage
    results = {
        'D-optimality': [],
        'A-optimality': [],
        'E-optimality': [],
        'Combined': [],
        'Count-Based': [],
        'Random': []
    }
    
    stats_storage = {
        'D-optimality': [],
        'A-optimality': [],
        'E-optimality': [],
        'Combined': [],
        'Count-Based': [],
        'Random': []
    }
    
    # Get environment dimensions
    test_env = gym.make(env_name)
    state_dim = int(np.prod(test_env.observation_space['image'].shape))
    action_dim = test_env.action_space.n
    test_env.close()
    
    print(f"\nEnvironment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Seeds: {n_seeds}, Episodes: {n_episodes}")
    print("\n" + "="*80)
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        set_seed(seed)
        
        # D-optimality
        env = gym.make(env_name)
        agent = SensorOptimizationAgent(state_dim, action_dim, optimality='D', 
                                       exploration_weight=exploration_weight)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['D-optimality'].append(np.mean(rewards[-20:]))
        stats_storage['D-optimality'].append(agent.get_statistics())
        env.close()
        print(f"  D-optimality: {results['D-optimality'][-1]:.3f}")
        
        # A-optimality
        env = gym.make(env_name)
        agent = SensorOptimizationAgent(state_dim, action_dim, optimality='A',
                                       exploration_weight=exploration_weight)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['A-optimality'].append(np.mean(rewards[-20:]))
        stats_storage['A-optimality'].append(agent.get_statistics())
        env.close()
        print(f"  A-optimality: {results['A-optimality'][-1]:.3f}")
        
        # E-optimality
        env = gym.make(env_name)
        agent = SensorOptimizationAgent(state_dim, action_dim, optimality='E',
                                       exploration_weight=exploration_weight)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['E-optimality'].append(np.mean(rewards[-20:]))
        stats_storage['E-optimality'].append(agent.get_statistics())
        env.close()
        print(f"  E-optimality: {results['E-optimality'][-1]:.3f}")
        
        # Combined
        env = gym.make(env_name)
        agent = SensorOptimizationAgent(state_dim, action_dim, optimality='combined',
                                       exploration_weight=exploration_weight)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['Combined'].append(np.mean(rewards[-20:]))
        stats_storage['Combined'].append(agent.get_statistics())
        env.close()
        print(f"  Combined: {results['Combined'][-1]:.3f}")
        
        # Count-Based
        env = gym.make(env_name)
        agent = CountBasedAgent(state_dim, action_dim, exploration_weight=exploration_weight)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['Count-Based'].append(np.mean(rewards[-20:]))
        env.close()
        print(f"  Count-Based: {results['Count-Based'][-1]:.3f}")
        
        # Random
        env = gym.make(env_name)
        agent = RandomAgent(state_dim, action_dim)
        rewards = train_agent(env, agent, n_episodes, max_steps, verbose=False, use_distance_reward=use_distance_reward)
        results['Random'].append(np.mean(rewards[-20:]))
        env.close()
        print(f"  Random: {results['Random'][-1]:.3f}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    summary = []
    for method in results.keys():
        mean_reward = np.mean(results[method])
        std_reward = np.std(results[method])
        summary.append((method, mean_reward, std_reward))
        print(f"{method:15s}: {mean_reward:.3f} ± {std_reward:.3f}")
    
    # Sort by mean reward
    summary.sort(key=lambda x: -x[1])
    
    print("\n" + "="*80)
    print("RANKING (by mean reward)")
    print("="*80)
    for i, (method, mean_reward, std_reward) in enumerate(summary, 1):
        print(f"{i}. {method:15s}: {mean_reward:.3f} ± {std_reward:.3f}")
    
    # Save results
    output_dir = "sensor_optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results_seed_{seed}.json", 'w') as f:
        json.dump({
            'results': {k: v for k, v in results.items()},
            'summary': summary
        }, f, indent=2)
    
    return results, summary


if __name__ == "__main__":
    # Run experiment
    results, summary = run_comparison_experiment(
        env_name="MiniGrid-Empty-5x5-v0",
        n_seeds=3,
        n_episodes=50,
        max_steps=50,
        exploration_weight=1.0,
        use_distance_reward=True
    )
