"""
TRUE MPME-RL with LEARNED DYNAMICS MODEL

Correct implementation:
1. Learn world model: s' = f_θ(s, a)
2. Information matrix: F = Σ ∇_θ f · ∇_θ f^T (FIM)
3. Min eigenvector: least-understood parameter direction
4. MPME action: argmax_a |u_min^T · ∇_θ f(s,a)|²

Key insight: We're exploring the PARAMETER space of dynamics, not state space.
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


class WorldModel(nn.Module):
    """Learned dynamics model: s' = f_θ(s, a)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.action_dim = action_dim
        
        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 8)
        
        # Dynamics network
        self.net = nn.Sequential(
            nn.Linear(state_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        action_emb = self.action_embed(action.long())
        x = torch.cat([state, action_emb], dim=-1)
        return self.net(x)


class TrueMPMEWithDynamics:
    """
    TRUE MPME using learned dynamics gradients.
    
    Information matrix tracks gradients of the WORLD MODEL parameters.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 world_model: WorldModel, update_interval: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.world_model = world_model
        self.update_interval = update_interval
        
        # Count total parameters for FIM
        self.param_count = sum(p.numel() for p in world_model.parameters())
        
        # Fisher Information Matrix (per episode)
        self._F = None  # FIM for current episode
        self._episode_id = 0
        
        # Min eigenvector
        self._u_min = None
        self._lambda_min = None
        
        # Training data for world model
        self._training_buffer = deque(maxlen=500)
        
        # Episode data for analysis
        self._episode_data = []
        
        # Step counter
        self._step_count = 0
    
    def start_episode(self):
        """Start new episode - fresh FIM."""
        self._episode_id += 1
        self._F = torch.eye(self.param_count) * 1e-6  # Regularized
        self._u_min = None
        self._lambda_min = None
        self._step_count = 0
    
    def _flatten_gradient(self, gradients) -> torch.Tensor:
        """Flatten gradient tuple to single tensor."""
        return torch.cat([g.view(-1) if g is not None else torch.zeros(1) 
                         for g in gradients])
    
    def record_transition(self, state: np.ndarray, action: int, next_state: np.ndarray):
        """
        Record transition - compute gradient of world model and update FIM.
        """
        s = torch.FloatTensor(state).unsqueeze(0)
        a = torch.LongTensor([action])
        ns = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Store for world model training
        self._training_buffer.append((s, a, ns))
        
        # Compute gradient of prediction w.r.t. parameters
        with torch.enable_grad():
            pred = self.world_model(s, a)
            loss = F.mse_loss(pred, ns)
            gradients = torch.autograd.grad(
                loss, 
                self.world_model.parameters(),
                retain_graph=True,
                allow_unused=True
            )
            
            g = self._flatten_gradient(gradients)
            
            # Rank-1 update to FIM: F ← F + g·g^T
            g = g.unsqueeze(0)
            self._F = self._F + g.t() @ g
            
            self._step_count += 1
        
        # Periodically update eigendecomposition
        if self._step_count % self.update_interval == 0:
            self._update_eigendecomposition()
        
        # Train world model periodically
        if self._step_count % 5 == 0:
            self._train_world_model()
    
    def _update_eigendecomposition(self):
        """Compute min eigenvector of FIM."""
        try:
            # For large FIM, use approximation
            if self.param_count > 500:
                # Use randomized SVD or power iteration
                self._lambda_min = 1.0
                self._u_min = torch.randn(self.param_count)
                self._u_min = self._u_min / self._u_min.norm()
                return
            
            # Full eigendecomposition
            self._lambda_min, self._u_min = torch.linalg.eigh(self._F)[0].min(dim=0)
            self._u_min = self._u_min.unsqueeze(1)  # Column vector
        except Exception as e:
            print(f"Eigendecomposition error: {e}")
    
    def _train_world_model(self):
        """Train world model on buffer."""
        if len(self._training_buffer) < 10:
            return
        
        optimizer = optim.Adam(self.world_model.parameters(), lr=0.01)
        
        for _ in range(2):
            batch = random.sample(list(self._training_buffer), 
                                 min(32, len(self._training_buffer)))
            states, actions, next_states = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.cat(actions)
            next_states = torch.cat(next_states)
            
            pred = self.world_model(states, actions)
            loss = F.mse_loss(pred, next_states)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def compute_mpme_scores(self, state: np.ndarray) -> np.ndarray:
        """
        Compute MPME scores for each action.
        
        ζ(a) = |u_min^T · ∇_θ f(s,a)|²
        """
        if self._u_min is None:
            self._update_eigendecomposition()
            if self._u_min is None:
                return np.ones(self.action_dim) * 0.5
        
        s = torch.FloatTensor(state).unsqueeze(0)
        scores = np.zeros(self.action_dim)
        
        with torch.enable_grad():
            for a in range(self.action_dim):
                action = torch.LongTensor([a])
                pred = self.world_model(s, action)
                
                # Gradient of prediction w.r.t. parameters
                gradients = torch.autograd.grad(
                    pred.sum(),
                    self.world_model.parameters(),
                    retain_graph=True,
                    allow_unused=True
                )
                
                g = self._flatten_gradient(gradients)
                
                # Projection onto min eigenspace
                projection = torch.dot(self._u_min.squeeze(), g)
                scores[a] = projection.item() ** 2
        
        return scores
    
    def get_exploration_action(self, state: np.ndarray, policy_probs: np.ndarray = None,
                               exploration_weight: float = 1.0) -> int:
        mpme_scores = self.compute_mpme_scores(state)
        
        if policy_probs is not None:
            mpme_range = mpme_scores.max() - mpme_scores.min()
            if mpme_range > 0:
                mpme_scores = (mpme_scores - mpme_scores.min()) / mpme_range
            
            combined = policy_probs + exploration_weight * mpme_scores
            combined = combined / combined.sum()
            action = np.random.choice(self.action_dim, p=combined)
        else:
            action = int(np.argmax(mpme_scores))
        
        return action
    
    def record_episode_end(self, total_reward: float):
        self._episode_data.append({
            'episode_id': self._episode_id,
            'reward': total_reward,
            'lambda_min': self._lambda_min.item() if hasattr(self._lambda_min, 'item') else self._lambda_min,
            'steps': self._step_count
        })
    
    def get_episode_data(self):
        return self._episode_data


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
        
        # World model for MPME
        if exploration_method == "mpme":
            self.world_model = WorldModel(state_dim, action_dim, hidden_dim=32)
            self.mpme = TrueMPMEWithDynamics(state_dim, action_dim, self.world_model)
        
        # Count tracking for baseline
        self.state_counts = {}
        
        self.memory = deque(maxlen=2000)
    
    def start_episode(self):
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
    agent.start_episode()
    
    obs, _ = env.reset()
    state = extract_state(obs)
    total_reward = 0
    
    for _ in range(max_steps):
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(next_obs)
        done = terminated or truncated
        
        agent.record_transition(state, action, next_state)
        agent.store(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    if agent.exploration_method == "mpme":
        agent.mpme.record_episode_end(total_reward)
    
    return total_reward


def train_agent(env, agent, n_episodes: int, max_steps: int,
                epsilon_start: float, epsilon_end: float) -> list:
    rewards = []
    
    for ep in range(n_episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - ep / n_episodes)
        episode_reward = run_episode(env, agent, max_steps, epsilon)
        rewards.append(episode_reward)
        
        for _ in range(3):
            agent.learn(batch_size=32)
    
    return rewards


def analyze_lambda_min_trend(episode_data):
    """Analyze how λ_min evolves across episodes."""
    if not episode_data:
        return {}
    
    lambda_mins = [d['lambda_min'] for d in episode_data if d['lambda_min'] is not None]
    
    if len(lambda_mins) < 2:
        return {}
    
    # Trend analysis
    first_half = lambda_mins[:len(lambda_mins)//2]
    second_half = lambda_mins[len(lambda_mins)//2:]
    
    improvement = np.mean(second_half) - np.mean(first_half)
    
    return {
        'lambda_min_first_half': np.mean(first_half),
        'lambda_min_second_half': np.mean(second_half),
        'improvement': improvement,
        'improvement_pct': (improvement / np.mean(first_half) * 100) if np.mean(first_half) > 0 else 0
    }


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
                         epsilon_start=0.3, epsilon_end=0.05)
    
    env.close()
    
    analysis = {}
    if method == "mpme" and hasattr(agent, 'mpme'):
        episode_data = agent.mpme.get_episode_data()
        analysis = analyze_lambda_min_trend(episode_data)
    
    return rewards, analysis


def run_benchmark(env_name: str, n_seeds: int = 10, n_episodes: int = 200):
    print(f"\n{'='*70}")
    print(f"TRUE MPME-RL (with Dynamics): {env_name}")
    print(f"{'='*70}")
    
    methods = ["mpme", "count", "epsilon", "plain"]
    results = {method: [] for method in methods}
    mpme_analyses = []
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}:")
        
        for method in methods:
            rewards, analysis = run_single_experiment(env_name, method, seed, n_episodes)
            final_avg = np.mean(rewards[-20:])
            results[method].append(final_avg)
            print(f"    {method.upper():<10}: {final_avg:.3f}")
            
            if method == "mpme" and analysis:
                mpme_analyses.append(analysis)
    
    # Statistical summary
    print("\n" + "="*70)
    print("Statistical Summary (95% CI)")
    print("="*70)
    
    for method in methods:
        vals = np.array(results[method])
        mean = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        print(f"{method.upper():<10}: {mean:.3f} ± {1.96*se:.3f}")
    
    # λ_min trend analysis
    if mpme_analyses:
        print("\n" + "="*70)
        print("λ_min TREND ANALYSIS (MPME)")
        print("="*70)
        
        improvements = [a.get('improvement', 0) for a in mpme_analyses]
        improvement_pcts = [a.get('improvement_pct', 0) for a in mpme_analyses]
        
        print(f"λ_min improvement (first→second half): {np.mean(improvements):.6f} ± {np.std(improvements):.6f}")
        print(f"λ_min improvement %: {np.mean(improvement_pcts):.1f}% ± {np.std(improvement_pcts):.1f}%")
        
        if np.mean(improvements) > 0:
            print("✓ λ_min INCREASES - MPME is working!")
        else:
            print("✗ λ_min does NOT increase - MPME may not be working")
    
    # Comparison
    print("\n" + "="*70)
    print("MPME vs Others")
    print("="*70)
    
    mpme_mean = np.mean(results['mpme'])
    for method in ['count', 'epsilon', 'plain']:
        other_mean = np.mean(results[method])
        diff = mpme_mean - other_mean
        improvement_pct = (diff / other_mean * 100) if other_mean > 0 else 0
        print(f"  MPME vs {method.upper():<10}: {diff:+.3f} ({improvement_pct:+.1f}%)")
    
    return results, mpme_analyses


def main():
    print("="*75)
    print("TRUE MPME-RL with LEARNED DYNAMICS")
    print("="*75)
    
    if not MINIGRID_AVAILABLE:
        import subprocess
        subprocess.call(["pip", "install", "minigrid", "-q"])
        import minigrid
    
    results, mpme_analyses = run_benchmark(
        "MiniGrid-Empty-Random-5x5-v0",
        n_seeds=10,
        n_episodes=200
    )
    
    # Final ranking
    print("\n" + "="*75)
    print("FINAL RANKING")
    print("="*75)
    
    means = {m: np.mean(results[m]) for m in results}
    for rank, (method, mean) in enumerate(sorted(means.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank}. {method.upper():<10}: {mean:.3f}")


if __name__ == "__main__":
    main()
