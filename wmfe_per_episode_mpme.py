"""
TRUE MPME-RL: Maximal Projection on Minimum Eigenspace

CORRECT understanding from MPME paper:

1. ORIGINAL MPME (sensor placement):
   - Φ̃ ∈ R^(N×n): signal representation matrix (all potential observation vectors)
   - Φ = HΦ̃: selected rows (sensor locations)
   - Ψ = Φ^T Φ: dual observation matrix
   - WCEV = σ²/λ_min(Ψ)
   - Goal: select sensors to minimize WCEV

2. RL MAPPING:
   - Each EPISODE is a different "game instance" with unique dynamics
   - Each (state, action, next_state) provides an "observation vector"
   - Per-episode matrix: Ψ_ep = Σ_{transitions} φ_i · φ_i^T
   - λ_min(Ψ_ep) measures "conditioning" of learned dynamics in THIS episode
   - MPME: select actions maximizing projection onto min eigenspace → improve λ_min

3. KEY INSIGHT:
   - Different episodes = different inverse problems
   - Need SEPARATE information matrix per episode
   - MPME guides exploration within each episode
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

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


class PerEpisodeMPME:
    """
    MPME with SEPARATE information matrix per episode.
    
    Each episode has its own "inverse problem" to solve.
    Transitions in episode E inform us about dynamics OF THAT EPISODE.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 feature_dim: int = 16, update_interval: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.update_interval = update_interval
        
        # === PER-EPISODE INFORMATION MATRIX ===
        # Each episode gets its own Ψ matrix
        self._current_psi = None  # Ψ = Σ φ_i · φ_i^T for current episode
        self._episode_id = 0
        
        # === FEATURE EXTRACTION ===
        # Maps transitions to "observation vectors" (like rows of Φ̃)
        self._transition_features = {}
        
        # === MINIMUM EIGENVECTOR TRACKING ===
        self._u_min = None
        self._lambda_min = None
        
        # === STATISTICS ===
        self._episode_stats = []  # Track λ_min per episode
    
    def _extract_transition_feature(self, state: np.ndarray, 
                                    action: int, 
                                    next_state: np.ndarray) -> np.ndarray:
        """
        Extract "observation vector" for a transition.
        
        In MPME, observation vectors are ROWS of signal representation matrix.
        Here, a transition (s, a, s') provides information about dynamics.
        
        Feature = [state, action_encoding, next_state, delta]
        """
        # State difference captures dynamics
        delta = next_state - state
        
        # Combine into feature vector
        feat = np.concatenate([
            state[:self.feature_dim//3],  # Current state snippet
            np.array([action / self.action_dim]),  # Action encoding
            next_state[:self.feature_dim//3],  # Next state snippet
            delta[:self.feature_dim//3]  # State change (dynamics!)
        ])[:self.feature_dim]
        
        # Normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        
        return feat
    
    def start_episode(self):
        """Start a new episode - initialize fresh Ψ matrix."""
        self._episode_id += 1
        self._current_psi = np.eye(self.feature_dim) * 0.001  # Regularized
        self._u_min = None
        self._lambda_min = None
    
    def record_transition(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """
        Record a transition - updates Ψ with rank-1 update.
        
        Ψ ← Ψ + φ · φ^T
        where φ is the observation vector for this transition.
        """
        # Extract observation vector for this transition
        phi = self._extract_transition_feature(state, action, next_state)
        
        # RANK-1 UPDATE: Ψ ← Ψ + φ · φ^T
        self._current_psi = self._current_psi + np.outer(phi, phi)
    
    def _compute_min_eigenvector(self):
        """
        Compute minimum eigenvector of Ψ.
        
        This is the direction of LEAST information about dynamics.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self._current_psi)
            idx = np.argsort(eigenvalues)
            self._lambda_min = eigenvalues[idx[0]]
            self._u_min = eigenvectors[:, idx[0]]
            return True
        except:
            return False
    
    def compute_mpme_scores(self, state: np.ndarray) -> np.ndarray:
        """
        Compute MPME exploration scores for candidate actions.
        
        For each action a:
            ζ(a) = |u_min^T · φ(s,a,?)|²
        
        But we don't know next_state yet! So we estimate:
            φ(s,a,?) ≈ [state, action, predicted_delta]
        
        The key insight: we project onto u_min to find actions that
        reveal dynamics in the WORST-MODELED direction.
        """
        if self._u_min is None:
            self._compute_min_eigenvector()
            if self._u_min is None:
                return np.ones(self.action_dim) * 0.5
        
        scores = np.zeros(self.action_dim)
        
        # Base state feature (common to all actions)
        state_feat = state[:self.feature_dim//3]
        
        for a in range(self.action_dim):
            # Construct estimated observation vector
            # φ(s,a) = [state, action, estimated_delta]
            # We use action encoding as proxy for expected delta
            action_encoding = np.array([a / self.action_dim])
            
            # Estimated feature (without knowing true next_state)
            phi_est = np.concatenate([
                state_feat,
                action_encoding,
                np.zeros(self.feature_dim - len(state_feat) - 1)
            ])[:self.feature_dim]
            
            # Normalize
            norm = np.linalg.norm(phi_est)
            if norm > 0:
                phi_est = phi_est / norm
            
            # MPME CRITERION: projection onto minimum eigenspace
            projection = np.dot(self._u_min, phi_est)
            scores[a] = projection ** 2
        
        return scores
    
    def get_exploration_action(self, state: np.ndarray,
                               policy_probs: np.ndarray = None,
                               exploration_weight: float = 1.0) -> int:
        """Get action using per-episode MPME."""
        mpme_scores = self.compute_mpme_scores(state)
        
        if policy_probs is not None:
            # Normalize and combine
            mpme_range = mpme_scores.max() - mpme_scores.min()
            if mpme_range > 0:
                mpme_scores = (mpme_scores - mpme_scores.min()) / mpme_range
            
            combined = policy_probs + exploration_weight * mpme_scores
            combined = combined / combined.sum()
            
            action = np.random.choice(self.action_dim, p=combined)
        else:
            action = int(np.argmax(mpme_scores))
        
        return action
    
    def get_episode_stats(self):
        """Get statistics for current episode."""
        return {
            'episode_id': self._episode_id,
            'lambda_min': self._lambda_min,
            'u_min': self._u_min.copy() if self._u_min is not None else None
        }
    
    def record_episode_stats(self):
        """Record stats at end of episode."""
        if self._lambda_min is not None:
            self._episode_stats.append(self._lambda_min)
    
    def get_all_episode_stats(self):
        """Get λ_min history across all episodes."""
        return self._episode_stats


class SimpleQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor):
        return self.net(state)


class RLAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 exploration_method: str = "mpme",
                 exploration_weight: float = 1.0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exploration_method = exploration_method
        self.exploration_weight = exploration_weight
        
        self.q_network = SimpleQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # PER-EPISODE MPME
        if exploration_method == "mpme":
            self.mpme = PerEpisodeMPME(state_dim, action_dim, feature_dim=16)
        
        # Count tracking
        self.state_counts = {}
        
        self.memory = deque(maxlen=2000)
    
    def start_episode(self):
        """Start new episode."""
        if self.exploration_method == "mpme":
            self.mpme.start_episode()
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        
        q_probs = F.softmax(q_values / 0.1, dim=-1).numpy()[0]
        
        if self.exploration_method == "mpme":
            action = self.mpme.get_exploration_action(state, q_probs, self.exploration_weight)
        elif self.exploration_method == "count":
            state_key = tuple(state.astype(np.int8)[:20])
            count = self.state_counts.get(state_key, 0)
            bonus = 1.0 / np.sqrt(count + 1)
            self.state_counts[state_key] = count + 1
            combined = q_probs + self.exploration_weight * bonus
            combined = combined / combined.sum()
            action = np.random.choice(self.action_dim, p=combined)
        else:
            action = q_values.argmax().item()
        
        return action
    
    def record_transition(self, state, action, next_state):
        """Record transition for MPME."""
        if self.exploration_method == "mpme":
            self.mpme.record_transition(state, action, next_state)
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch_size: int = 32, gamma: float = 0.99):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(q_values.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def extract_state(obs):
    if isinstance(obs, dict):
        return obs['image'].flatten()
    return obs.flatten() if hasattr(obs, 'flatten') else obs


def run_episode(env, agent, max_steps: int, epsilon: float = 0.0) -> float:
    """Run one episode."""
    agent.start_episode()  # Initialize per-episode MPME
    
    obs, _ = env.reset()
    state = extract_state(obs)
    total_reward = 0
    
    for _ in range(max_steps):
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(next_obs)
        done = terminated or truncated
        
        # Record for MPME
        agent.record_transition(state, action, next_state)
        
        # Store for learning
        agent.store(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return total_reward


def train_agent(env, agent, n_episodes: int, max_steps: int,
                epsilon_start: float, epsilon_end: float,
                print_every: int = 50) -> list:
    rewards = []
    
    for ep in range(n_episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - ep / n_episodes)
        episode_reward = run_episode(env, agent, max_steps, epsilon)
        rewards.append(episode_reward)
        
        for _ in range(3):
            agent.learn(batch_size=32)
        
        if print_every > 0 and (ep + 1) % print_every == 0:
            avg_r = np.mean(rewards[-print_every:])
            print(f"    Episodes {ep-print_every+2}-{ep+1}: Avg Reward={avg_r:.3f}")
    
    return rewards


def run_single_experiment(env_name: str, method: str, seed: int,
                          n_episodes: int = 200, max_steps: int = 100):
    set_seed(seed)
    
    env = gym.make(env_name)
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = int(np.prod(env.observation_space['image'].shape))
    else:
        state_dim = int(np.prod(env.observation_space.shape))
    
    action_dim = env.action_space.n
    
    agent = RLAgent(state_dim, action_dim, method, exploration_weight=1.0)
    rewards = train_agent(env, agent, n_episodes, max_steps,
                         epsilon_start=0.3, epsilon_end=0.05, print_every=0)
    
    env.close()
    
    # Get MPME stats
    stats = None
    if method == "mpme" and hasattr(agent, 'mpme'):
        stats = {
            'lambda_min_history': agent.mpme.get_all_episode_stats()
        }
    
    return rewards, stats


def run_benchmark(env_name: str, n_seeds: int = 10, n_episodes: int = 200):
    print(f"\n{'='*70}")
    print(f"PER-EPISODE MPME-RL Benchmark: {env_name}")
    print(f"{'='*70}")
    
    methods = ["mpme", "count", "epsilon", "plain"]
    results = {method: [] for method in methods}
    mpme_lambda_history = []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")
        
        for method in methods:
            rewards, stats = run_single_experiment(env_name, method, seed, n_episodes)
            final_avg = np.mean(rewards[-20:])
            results[method].append(final_avg)
            print(f"    {method.upper():<10}: {final_avg:.3f}")
            
            if method == "mpme" and stats:
                if stats['lambda_min_history']:
                    mpme_lambda_history.extend(stats['lambda_min_history'])
    
    # Statistical analysis
    print("\n" + "="*70)
    print("Statistical Analysis (95% CI)")
    print("="*70)
    
    for method in methods:
        vals = np.array(results[method])
        mean = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        print(f"{method.upper():<10}: {mean:.3f} ± {1.96*se:.3f}")
    
    # Pairwise comparison (simplified without scipy)
    print("\n" + "="*70)
    print("MPME vs Others (Mean Difference)")
    print("="*70)
    
    mpme_vals = np.array(results['mpme'])
    for method in ['count', 'epsilon', 'plain']:
        other_vals = np.array(results[method])
        diff = mpme_vals.mean() - other_vals.mean()
        improvement_pct = (diff / other_vals.mean() * 100) if other_vals.mean() > 0 else 0
        print(f"  MPME vs {method.upper():<10}: diff={diff:+.3f}, improvement={improvement_pct:+.1f}%")
    
    # MPME λ_min analysis
    if mpme_lambda_history:
        print("\n" + "="*70)
        print("MPME λ_min Analysis (per-episode)")
        print("="*70)
        lam_min = np.array(mpme_lambda_history)
        print(f"Mean λ_min: {lam_min.mean():.6f}")
        print(f"Std λ_min: {lam_min.std():.6f}")
        print(f"Min λ_min: {lam_min.min():.6f}")
        print(f"Max λ_min: {lam_min.max():.6f}")
    
    return results


def main():
    print("="*75)
    print("PER-EPISODE MPME-RL: Maximal Projection on Minimum Eigenspace")
    print("Correct Implementation: Separate FIM per Episode")
    print("="*75)
    
    if not MINIGRID_AVAILABLE:
        import subprocess
        subprocess.call(["pip", "install", "minigrid", "-q"])
        import minigrid
    
    results = run_benchmark("MiniGrid-Empty-Random-5x5-v0", n_seeds=5, n_episodes=200)
    
    # Final ranking
    print("\n" + "="*75)
    print("FINAL RANKING")
    print("="*75)
    
    means = {m: np.mean(results[m]) for m in results}
    for rank, (method, mean) in enumerate(sorted(means.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank}. {method.upper():<10}: {mean:.3f}")


if __name__ == "__main__":
    main()
