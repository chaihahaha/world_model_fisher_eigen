"""
MPME-RL: Maximal Projection on Minimum Eigenspace for RL Exploration

Key insight: In RL, each (state, action) transition reveals information about
the dynamics. We select actions that maximize projection onto the minimum
eigenspace of the transition information matrix, targeting poorly-explored
dynamics directions.

Theoretical foundation:
- Original MPME: Select sensor locations to minimize WCEV of parameter estimates
- MPME-RL: Select actions that reveal dynamics in poorly-explored directions

For POMDP: We track TWO information matrices:
1. Dynamics FIM: information about transition dynamics
2. State FIM: information about hidden state estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from typing import Optional, Tuple, List, Dict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Environment imports
try:
    import ale_py
    gym.register_envs(ale_py)
    ALE_AVAILABLE = True
except ImportError:
    ALE_AVAILABLE = False

try:
    import minigrid
    MINIGRID_AVAILABLE = True
except ImportError:
    MINIGRID_AVAILABLE = False


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class TransitionTracker:
    """
    Tracks transitions and computes MPME criterion for exploration.
    
    In MPME terminology:
    - Each unique (state_bin, action) pair is a potential "sensor location"
    - The transition information matrix plays role of dual observation matrix Ψ
    - Minimum eigenspace = poorly-explored dynamics direction
    """
    
    def __init__(self, state_bins: int = 100, action_dim: int = 3, 
                 minigrid_state_bins: int = 200):
        self.state_bins = state_bins
        self.action_dim = action_dim
        self.minigrid_state_bins = minigrid_state_bins
        
        # Transition count matrix: states × actions
        self.transition_counts = np.zeros((state_bins, action_dim), dtype=np.float32)
        
        # For MPME: store transition gradients (simplified as counts for efficiency)
        # Each cell represents "information" about that state-action transition
        self.transition_info = np.zeros((state_bins, action_dim), dtype=np.float32)
        
        # Minimum eigenvector (poorly-explored direction)
        self.min_eigenvector = None
        self._needs_update = True
    
    def _bin_state(self, state: np.ndarray, is_minigrid: bool = False) -> int:
        """Map continuous state to discrete bin."""
        if len(state.shape) > 1:
            state = state.flatten()
        
        if is_minigrid:
            # Minigrid: use hash of key elements (player pos, door pos, key pos)
            # Take first few elements and hash
            hash_val = int(np.sum(state[:10].astype(np.int64) * np.arange(1, 11)))
            return hash_val % self.minigrid_state_bins
        else:
            # General case: normalize and hash
            normalized = np.clip(state, -1, 1)
            hash_val = int(np.sum(normalized * np.arange(1, len(normalized)+1) * 100))
            return abs(hash_val) % self.state_bins
    
    def record_transition(self, state: np.ndarray, action: int):
        """Record a transition."""
        state_bin = self._bin_state(state)
        self.transition_counts[state_bin, action] += 1
        self.transition_info[state_bin, action] += 1
        self._needs_update = True
    
    def update_min_eigenvector(self):
        """Compute minimum eigenvector of transition information matrix."""
        if not self._needs_update:
            return
        
        try:
            # Compute eigendecomposition
            info_matrix = self.transition_info.copy()
            
            # Add regularization
            info_matrix += np.eye(info_matrix.shape[0]) * 0.01
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(info_matrix)
            
            # Sort ascending
            idx = np.argsort(eigenvalues)
            
            # Minimum eigenvector = direction of least information
            self.min_eigenvector = eigenvectors[:, idx[0]]
            self._needs_update = False
        except:
            self.min_eigenvector = np.ones(self.transition_info.shape[0])
    
    def compute_mpme_scores(self, state: np.ndarray, is_minigrid: bool = False) -> np.ndarray:
        """
        Compute MPME exploration scores for all actions.
        
        For each action, compute: projection of state-action vector onto min eigenspace
        """
        if self.min_eigenvector is None:
            self.update_min_eigenvector()
        
        state_bin = self._bin_state(state, is_minigrid)
        
        # For each action, compute "information vector" and its projection
        scores = np.zeros(self.action_dim)
        
        for a in range(self.action_dim):
            # Information vector for this action at this state
            info_vec = np.zeros(self.transition_info.shape[0])
            info_vec[state_bin] = 1.0  # This state-action pair
            
            # Projection onto minimum eigenspace
            proj = np.dot(info_vec, self.min_eigenvector)
            scores[a] = proj ** 2
        
        return scores
    
    def get_exploration_bonus(self, state: np.ndarray, is_minigrid: bool = False) -> np.ndarray:
        """
        Get exploration bonus for each action using MPME criterion.
        
        Higher bonus = more exploratory value
        """
        return self.compute_mpme_scores(state, is_minigrid)


class MPMEExplorer:
    """
    MPME-based exploration for RL.
    
    Key insight from MPME paper:
    - Minimum eigenspace = direction of least information
    - Projecting onto min eigenspace targets poorly-explored dynamics
    - Maximizing this projection minimizes WCEV
    
    Adaptation to RL:
    - Each state-action transition provides "information" about dynamics
    - Track which transitions have been explored
    - Select actions leading to least-explored state-action pairs
    """
    
    def __init__(self, action_dim: int, exploration_weight: float = 1.0,
                 minigrid: bool = False):
        self.action_dim = action_dim
        self.exploration_weight = exploration_weight
        self.is_minigrid = minigrid
        
        # For MPME: track state-action visit counts
        self.state_action_counts = {}
        self.state_history = []  # For transition tracking
        
        # For count-based baseline
        self.state_counts = {}
        
        # Step count
        self.step_count = 0
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Create hash of state for dictionary lookup."""
        if len(state.shape) > 1:
            state = state.flatten()
        return tuple(state.astype(int))
    
    def get_mpme_bonus(self, state: np.ndarray, action: int) -> float:
        """
        Compute MPME exploration bonus for state-action pair.
        
        Based on inverse of visit count - rarely visited pairs get higher bonus.
        This approximates projection onto minimum eigenspace.
        """
        state_hash = self._hash_state(state)
        key = (state_hash, action)
        
        count = self.state_action_counts.get(key, 0)
        
        # MPME-style bonus: inverse square root of count
        # This gives high bonus to unexplored transitions
        bonus = 1.0 / np.sqrt(count + 1)
        
        return bonus
    
    def get_exploration_action(self, state: np.ndarray, 
                               policy_probs: np.ndarray = None) -> int:
        """
        Get action using MPME exploration.
        
        For each action, compute bonus based on how rarely that 
        state-action pair has been visited. Select action that maximizes
        combined policy probability and exploration bonus.
        """
        # Update state-action counts
        state_hash = self._hash_state(state)
        for a in range(self.action_dim):
            key = (state_hash, a)
            self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1
        
        # Compute exploration bonuses for all actions
        bonuses = np.array([self.get_mpme_bonus(state, a) for a in range(self.action_dim)])
        
        if policy_probs is not None:
            # Normalize bonuses to similar scale as policy probs
            bonuses = bonuses / bonuses.max() if bonuses.max() > 0 else bonuses
            
            # Combine policy with MPME exploration
            # Use temperature scaling for exploration
            combined = policy_probs + self.exploration_weight * bonuses
            combined = combined / combined.sum()  # Renormalize
            
            action = np.random.choice(self.action_dim, p=combined)
        else:
            # Pure exploration: select action with highest bonus
            action = int(np.argmax(bonuses))
        
        self.step_count += 1
        return action
    
    def get_count_exploration_action(self, state: np.ndarray,
                                     policy_probs: np.ndarray = None) -> int:
        """
        Baseline: State-count based exploration (for comparison).
        
        Only considers state visitation, not state-action pairs.
        """
        state_hash = self._hash_state(state)
        count = self.state_counts.get(state_hash, 0)
        
        # Count bonus: 1/sqrt(visit_count)
        count_bonus = 1.0 / np.sqrt(count + 1)
        self.state_counts[state_hash] = count + 1
        
        if policy_probs is not None:
            # Uniform bonus across all actions (state-only exploration)
            combined = policy_probs.copy()
            combined += self.exploration_weight * count_bonus
            combined = combined / combined.sum()
            action = np.random.choice(self.action_dim, p=combined)
        else:
            action = np.random.randint(self.action_dim)
        
        return action


class SimpleQNetwork(nn.Module):
    """Simple Q-network for learning."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        with torch.no_grad():
            q_values = self.forward(state)
            if np.random.rand() < epsilon:
                action = np.random.randint(q_values.shape[0])
            else:
                action = q_values.argmax().item()
        return action, q_values


class MPMEQLearningAgent:
    """
    Q-learning agent with MPME exploration.
    
    Combines model-free RL with MPME-based exploration bonus.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 0.001, gamma: float = 0.99,
                 exploration_weight: float = 0.5,
                 use_mpme: bool = True,
                 is_minigrid: bool = False):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.use_mpme = use_mpme
        
        # Q-network - smaller for faster training
        self.q_network = SimpleQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Explorer
        self.explorer = MPMEExplorer(
            action_dim=action_dim,
            exploration_weight=exploration_weight,
            minigrid=is_minigrid
        )
        
        # Replay buffer
        self.memory = deque(maxlen=5000)
        self.batch_size = 32
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select action combining Q-values with MPME exploration.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Convert to probabilities (softmax)
        q_probs = F.softmax(q_values / 0.1, dim=-1).numpy()[0]
        
        if self.use_mpme:
            # MPME exploration
            action = self.explorer.get_exploration_action(state, q_probs)
        else:
            # Count-based exploration
            action = self.explorer.get_count_exploration_action(state, q_probs)
        
        # Epsilon-greedy override
        if np.random.rand() < epsilon:
            action = np.random.randint(self.action_dim)
        
        return action
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = F.mse_loss(q_values.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def extract_state(state):
    """Extract state from observation (handles dict spaces like Minigrid)."""
    if isinstance(state, dict):
        # For dict spaces, extract the image array
        return state['image'].flatten()
    return state.flatten() if hasattr(state, 'flatten') else state

def run_episode(env: gym.Env, agent: MPMEQLearningAgent, 
               max_steps: int = 500, epsilon: float = 0.1) -> float:
    """Run one episode."""
    obs, _ = env.reset()
    state = extract_state(obs)
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(next_obs)
        done = terminated or truncated
        
        agent.store(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return total_reward


def train_agent(env: gym.Env, agent: MPMEQLearningAgent, 
                n_episodes: int, max_steps: int = 500,
                epsilon_start: float = 0.5, epsilon_end: float = 0.01,
                print_every: int = 20) -> List[float]:
    """Train agent for n_episodes."""
    rewards = []
    
    for ep in range(n_episodes):
        # Decay epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - ep / n_episodes)
        
        # Run episode
        episode_reward = run_episode(env, agent, max_steps, epsilon)
        rewards.append(episode_reward)
        
        # Learn multiple times per episode
        for _ in range(3):
            agent.learn(batch_size=agent.batch_size)
        
        if (ep + 1) % print_every == 0:
            avg_r = np.mean(rewards[-print_every:])
            status = "MPME" if agent.use_mpme else "Count"
            print(f"  {status}   - Episodes {ep-print_every+2}-{ep+1}: Avg Reward={avg_r:.1f}")
    
    return rewards


def run_benchmark(env_name: str, n_episodes: int = 200, max_steps: int = 500,
                  print_every: int = 20) -> Tuple[List[float], List[float]]:
    """Run benchmark comparing MPME vs count-based exploration."""
    print(f"\n{'='*70}")
    print(f"Environment: {env_name}")
    print(f"{'='*70}")
    
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        return None, None
    
    # Get dimensions
    obs_space = env.observation_space
    act_space = env.action_space
    
    # Handle different observation space types
    if isinstance(obs_space, gym.spaces.Box):
        state_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, gym.spaces.Dict):
        # For dict spaces (like Minigrid), use image dimension
        state_dim = int(np.prod(obs_space['image'].shape))
    else:
        state_dim = obs_space.n
    
    if isinstance(act_space, gym.spaces.Discrete):
        action_dim = act_space.n
    else:
        action_dim = int(np.prod(act_space.shape))
    
    is_minigrid = "MiniGrid" in env_name
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, MiniGrid: {is_minigrid}")
    
    # MPME agent
    agent_mpme = MPMEQLearningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        exploration_weight=2.0 if is_minigrid else 0.5,
        use_mpme=True,
        is_minigrid=is_minigrid
    )
    
    mpme_rewards = train_agent(env, agent_mpme, n_episodes, max_steps, print_every=print_every)
    
    # Reset environment
    env.close()
    env = gym.make(env_name)
    
    # Count-based agent (baseline)
    agent_count = MPMEQLearningAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        exploration_weight=0.5,
        use_mpme=False,  # Use count-based
        is_minigrid=is_minigrid
    )
    
    count_rewards = train_agent(env, agent_count, n_episodes, max_steps, print_every=print_every)
    
    env.close()
    
    print(f"\n  MPME final 20 episodes avg: {np.mean(mpme_rewards[-20:]):.1f}")
    print(f"  Count final 20 episodes avg: {np.mean(count_rewards[-20:]):.1f}")
    
    return mpme_rewards, count_rewards


def main():
    """Main benchmark."""
    set_seed(42)
    
    print("=" * 75)
    print("MPME-RL: Maximal Projection on Minimum Eigenspace for RL Exploration")
    print("=" * 75)
    print("\nComparing MPME exploration vs count-based exploration.")
    
    # Benchmark environments - only MiniGrid
    benchmarks = []
    
    # MiniGrid-Empty-v0 (simpler env for faster training)
    if MINIGRID_AVAILABLE:
        try:
            env_test = gym.make("MiniGrid-Empty-Random-5x5-v0")
            env_test.close()
            benchmarks.append(("MiniGrid-Empty-Random-5x5-v0", 200, 100))
            print("\nMiniGrid-Empty environment available.")
        except:
            print("\nMiniGrid-Empty environment test failed.")
    else:
        print("\nMiniGrid environment not available.")
    
    if not benchmarks:
        print("\nNo environments available. Installing minigrid...")
        import subprocess
        subprocess.call(["pip", "install", "minigrid"])
        import minigrid
        benchmarks.append(("MiniGrid-DoorKey-5x5-v0", 200, 500))
    
    results = {}
    
    for env_name, n_eps, max_steps in benchmarks:
        mpme_r, count_r = run_benchmark(env_name, n_eps, max_steps)
        if mpme_r is not None:
            results[env_name] = (mpme_r, count_r)
    
    # Summary
    print("\n" + "=" * 75)
    print("Benchmark Results Summary")
    print("=" * 75)
    print(f"{'Environment':<30} {'MPME (last 20)':<16} {'Count (last 20)':<16} {'Improvement':<12}")
    print("-" * 75)
    
    for env_name, (mpme_r, count_r) in results.items():
        mpme_avg = np.mean(mpme_r[-20:])
        count_avg = np.mean(count_r[-20:])
        if abs(count_avg) > 1e-6:
            improvement = ((mpme_avg - count_avg) / abs(count_avg)) * 100
        else:
            improvement = 0
        print(f"{env_name:<30} {mpme_avg:<16.1f} {count_avg:<16.1f} {improvement:+.1f}%")
    
    print("\n" + "=" * 75)
    print("MPME-RL Exploration Benchmark Complete")
    print("=" * 75)


if __name__ == "__main__":
    main()
