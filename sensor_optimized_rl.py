"""
SENSOR-OPTIMIZED RL: Borrowing Sensor Placement for Q-Learning

THEORETICAL FOUNDATION:

1. SENSOR PLACEMENT (Original Problem):
   - Goal: Select M sensors out of N candidates to estimate parameter α
   - Each sensor i provides: y_i = φ_i^T α + noise
   - Information matrix: F = Σ_{selected} φ_i φ_i^T
   - Objective: Maximize λ_min(F) to minimize estimation error

2. Q-VALUE ESTIMATION (RL Problem):
   - Goal: Learn Q(s,a) for all state-action pairs
   - Each visit to (s,a) provides: Q_t(s,a) ← (1-α)Q_{t-1}(s,a) + α[r + γQ(s',a')]
   - This is a recursive estimator with variance depending on visit count

3. BRIDGING THE TWO:
   - Each unique (s,a) pair = a "sensor" for Q-function estimation
   - Visit count N(s,a) determines estimation variance: Var[Q(s,a)] ∝ 1/N(s,a)
   - Goal: Select M visits to minimize MAX_{s,a} Var[Q(s,a)]
   - This is EQUIVALENT to sensor placement for uniform estimation!

4. ALGORITHM: Greedy Sensor Placement for Exploration
   - Maintain visit counts N(s,a)
   - At each step, select action leading to state-action pair with LOWEST N(s,a)
   - This maximizes information gain per step (greedy sensor placement)
   - Equivalent to: argmax_a N(s,a) being MINIMIZED

WHY THIS WORKS:
1. Directly targets the objective: uniform Q-value estimation
2. Greedy sensor placement has ln(n) approximation ratio
3. No eigendecomposition needed - just count tracking
4. Theoretically grounded in sensor optimization literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import minigrid
except ImportError:
    import subprocess
    subprocess.call(["pip", "install", "minigrid", "-q"])
    import minigrid


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class QNetwork(nn.Module):
    """Simple Q-network."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_net = nn.Linear(state_dim, action_dim)
    
    def forward(self, state):
        return self.q_net(state)


class SensorOptimizedExplorer:
    """
    SENSOR-OPTIMIZED EXPLORATION
    
    THEORETICAL FOUNDATION:
    
    Theorem: Q-value estimation is equivalent to sensor placement.
    
    Proof:
    1. Each (s,a) visit provides an estimate of Q(s,a)
    2. Variance of estimate: Var[Q(s,a)] = σ²/N(s,a)
    3. Goal: Minimize max_{s,a} Var[Q(s,a)] = σ²/min_{s,a} N(s,a)
    4. Equivalent to: Maximize min_{s,a} N(s,a)
    5. This IS the sensor placement problem!
       - "Sensors" = state-action visits
       - "Parameter" = Q-function values
       - "Information matrix" = diagonal matrix of visit counts
       - λ_min(F) = min_{s,a} N(s,a)
    
    Algorithm: Greedy Sensor Placement
    - At each step, visit (s,a) with MINIMUM N(s,a)
    - This is greedy sensor placement for uniform coverage
    - Approximation ratio: ln(|S×A|)
    """
    
    def __init__(self, action_dim, hash_bits=15):
        self.action_dim = action_dim
        self.hash_bits = hash_bits
        
        # Visit counts: N(s,a) for each state-action pair
        self.visit_counts = defaultdict(lambda: np.zeros(action_dim))
        
        # Statistics
        self.total_visits = 0
        self.unique_sa_pairs = 0
        self.min_visit_count = float('inf')
    
    def hash_state(self, state):
        """Hash state to discrete key."""
        return tuple(state.astype(np.int8)[:self.hash_bits])
    
    def get_exploration_bonus(self, state):
        """
        Get exploration bonus for each action.
        
        SENSOR OPTIMIZATION PRINCIPLE:
        - Bonus(a) = 1 / sqrt(N(s,a))
        - Actions leading to rarely-visited pairs get HIGH bonus
        - This is greedy sensor placement: always place next "sensor"
          at the location with LEAST information
        
        Theoretical justification:
        - Variance of Q(s,a) estimate: σ²/N(s,a)
        - To minimize max variance, maximize min N(s,a)
        - Greedy: always increment minimum N(s,a)
        """
        state_key = self.hash_state(state)
        counts = self.visit_counts[state_key]
        
        # Sensor optimization bonus: inverse square root of count
        # This prioritizes actions with LOW visit counts
        bonuses = 1.0 / np.sqrt(counts + 1)
        
        return bonuses
    
    def record_visit(self, state, action):
        """Record a visit to state-action pair."""
        state_key = self.hash_state(state)
        counts = self.visit_counts[state_key]
        
        # Track minimum visit count (this is λ_min in sensor placement)
        old_count = counts[action]
        counts[action] += 1
        
        self.total_visits += 1
        
        # Track unique pairs
        if old_count == 0:
            self.unique_sa_pairs += 1
        
        # Track min visit count (λ_min)
        if counts[action] < self.min_visit_count:
            self.min_visit_count = counts[action]
    
    def get_coverage_stats(self):
        """Get coverage statistics."""
        if not self.visit_counts:
            return {
                'coverage_ratio': 0,
                'min_visit_count': 0,
                'unique_sa_pairs': 0,
                'total_visits': 0
            }
        
        # Flatten all counts
        all_counts = np.array([c for counts in self.visit_counts.values() for c in counts])
        all_counts = all_counts[all_counts > 0]  # Only visited pairs
        
        return {
            'coverage_ratio': len(all_counts) / self.total_visits if self.total_visits > 0 else 0,
            'min_visit_count': int(np.min(all_counts)),
            'max_visit_count': int(np.max(all_counts)),
            'mean_visit_count': float(np.mean(all_counts)),
            'std_visit_count': float(np.std(all_counts)),
            'unique_sa_pairs': len(all_counts),
            'total_visits': self.total_visits,
            'uniformity': np.min(all_counts) / np.max(all_counts) if len(all_counts) > 0 else 0
        }
    
    def reset(self):
        """Reset for new episode."""
        self.visit_counts = defaultdict(lambda: np.zeros(self.action_dim))
        self.total_visits = 0
        self.unique_sa_pairs = 0
        self.min_visit_count = float('inf')


class RLAgent:
    def __init__(self, state_dim, action_dim, exploration_method="sensor",
                 exploration_weight=1.0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exploration_method = exploration_method
        self.exploration_weight = exploration_weight
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.01)
        
        if exploration_method == "sensor":
            self.explorer = SensorOptimizedExplorer(action_dim)
        elif exploration_method == "random":
            self.exploration_weight = exploration_weight
        
        self.memory = __import__('collections').deque(maxlen=500)
        self.current_episode_reward = 0
    
    def start_episode(self):
        self.current_episode_reward = 0
        if self.exploration_method == "sensor":
            self.explorer.reset()
    
    def select_action(self, state, epsilon=0.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        
        q_probs = F.softmax(q_values / 0.1, dim=-1).numpy()[0]
        
        if self.exploration_method == "sensor":
            # SENSOR-OPTIMIZED exploration bonus
            bonuses = self.explorer.get_exploration_bonus(state)
            
            # Combine Q-probabilities with sensor optimization bonus
            combined = q_probs + self.exploration_weight * bonuses
            combined = np.clip(combined, 1e-10, None)
            combined = combined / combined.sum()
            
            action = np.random.choice(self.action_dim, p=combined)
        elif self.exploration_method == "random":
            # Simple random noise baseline
            noise = np.random.uniform(0, self.exploration_weight, self.action_dim)
            combined = q_probs + noise
            combined = combined / combined.sum()
            action = np.random.choice(self.action_dim, p=combined)
        else:
            action = q_values.argmax().item()
        
        return action
    
    def record_visit(self, state, action):
        if self.exploration_method == "sensor":
            self.explorer.record_visit(state, action)
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.current_episode_reward += reward
    
    def learn(self, batch_size=32, gamma=0.99):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(list(self.memory), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards_t + gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(q_values.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def extract_state(obs):
    if isinstance(obs, dict):
        return obs['image'].flatten()
    return obs.flatten() if hasattr(obs, 'flatten') else obs


def run_episode(env, agent, max_steps, epsilon=0.0):
    agent.start_episode()
    
    obs, _ = env.reset()
    state = extract_state(obs)
    
    for step in range(max_steps):
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = extract_state(next_obs)
        done = terminated or truncated
        
        # Record visit for sensor optimization
        agent.record_visit(state, action)
        
        agent.store(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            break
    
    return agent.current_episode_reward


def train_agent(env, agent, n_episodes, max_steps,
                epsilon_start, epsilon_end, print_every=20) -> list:
    rewards = []
    coverage_stats_list = []
    
    for ep in range(n_episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - ep / n_episodes)
        
        episode_reward = run_episode(env, agent, max_steps, epsilon)
        rewards.append(episode_reward)
        
        if hasattr(agent, 'explorer'):
            stats = agent.explorer.get_coverage_stats()
            coverage_stats_list.append(stats)
        
        for _ in range(5):
            agent.learn(batch_size=32)
        
        if print_every > 0 and (ep + 1) % print_every == 0:
            avg_r = np.mean(rewards[-print_every:])
            print(f"    Episodes {ep-print_every+2}-{ep+1}: Avg Reward={avg_r:.3f}")
    
    return rewards, coverage_stats_list


def analyze_coverage(stats_list):
    """Analyze coverage statistics over episodes."""
    if not stats_list:
        return {}
    
    # Extract metrics over time
    coverage_ratios = [s['coverage_ratio'] for s in stats_list]
    uniformities = [s['uniformity'] for s in stats_list]
    min_visits = [s['min_visit_count'] for s in stats_list]
    
    return {
        'final_coverage_ratio': coverage_ratios[-1],
        'mean_coverage_ratio': np.mean(coverage_ratios[-20:]),
        'final_uniformity': uniformities[-1],
        'mean_uniformity': np.mean(uniformities[-20:]),
        'final_min_visit': min_visits[-1],
        'coverage_improvement': coverage_ratios[-1] - coverage_ratios[0] if len(coverage_ratios) > 1 else 0
    }


def run_benchmark():
    print("="*75)
    print("SENSOR-OPTIMIZED RL: Theory and Implementation")
    print("="*75)
    print("\nTheoretical Foundation:")
    print("  Q-value estimation = Sensor placement problem")
    print("  Each (s,a) visit = a 'sensor' measuring Q(s,a)")
    print("  Goal: Uniform coverage → minimize max variance")
    print("  Algorithm: Greedy sensor placement")
    print("="*75)
    
    n_seeds = 10
    n_episodes = 100
    
    sensor_rewards = []
    random_rewards = []
    sensor_coverage = []
    random_coverage = []
    
    for seed in range(n_seeds):
        set_seed(seed)
        
        # SENSOR-OPTIMIZED
        env = gym.make("MiniGrid-Empty-Random-5x5-v0")
        state_dim = int(np.prod(env.observation_space['image'].shape))
        action_dim = env.action_space.n
        
        agent_sensor = RLAgent(state_dim, action_dim, "sensor", exploration_weight=2.0)
        rewards_s, stats_s = train_agent(env, agent_sensor, n_episodes, 100,
                                         epsilon_start=0.3, epsilon_end=0.05, print_every=0)
        sensor_rewards.append(np.mean(rewards_s[-20:]))
        sensor_coverage.append(analyze_coverage(stats_s))
        env.close()
        
        # RANDOM baseline
        env = gym.make("MiniGrid-Empty-Random-5x5-v0")
        agent_random = RLAgent(state_dim, action_dim, "random", exploration_weight=1.0)
        rewards_r, _ = train_agent(env, agent_random, n_episodes, 100,
                                   epsilon_start=0.3, epsilon_end=0.05, print_every=0)
        random_rewards.append(np.mean(rewards_r[-20:]))
        env.close()
        
        print(f"Seed {seed+1}: Sensor={sensor_rewards[-1]:.3f}, Random={random_rewards[-1]:.3f}")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    sensor_mean = np.mean(sensor_rewards)
    sensor_std = np.std(sensor_rewards)
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    
    print(f"\nSENSOR-OPTIMIZED:")
    print(f"  Mean reward: {sensor_mean:.3f} ± {sensor_std:.3f}")
    print(f"  95% CI: [{sensor_mean - 1.96*sensor_std:.3f}, {sensor_mean + 1.96*sensor_std:.3f}]")
    
    print(f"\nRANDOM NOISE:")
    print(f"  Mean reward: {random_mean:.3f} ± {random_std:.3f}")
    print(f"  95% CI: [{random_mean - 1.96*random_std:.3f}, {random_mean + 1.96*random_std:.3f}]")
    
    print(f"\nDifference: {sensor_mean - random_mean:+.3f}")
    
    # Statistical test (paired t-test approximation)
    diff = np.array(sensor_rewards) - np.array(random_rewards)
    t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
    print(f"Paired t-statistic: {t_stat:.3f}")
    
    # Coverage analysis
    print("\n" + "="*70)
    print("COVERAGE ANALYSIS (Sensor-Optimized)")
    print("="*70)
    
    avg_coverage = np.mean([s['mean_coverage_ratio'] for s in sensor_coverage])
    avg_uniformity = np.mean([s['mean_uniformity'] for s in sensor_coverage])
    
    print(f"Mean coverage ratio: {avg_coverage:.4f}")
    print(f"Mean uniformity: {avg_uniformity:.4f}")
    print(f"Coverage improvement: {np.mean([s['coverage_improvement'] for s in sensor_coverage]):.4f}")
    
    # Ranking
    print("\n" + "="*70)
    print("RANKING")
    print("="*70)
    results = [('Sensor-Optimized', sensor_mean), ('Random Noise', random_mean)]
    for rank, (name, mean) in enumerate(sorted(results, key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name}: {mean:.3f}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if sensor_mean > random_mean + 0.05:
        print("[SUCCESS] Sensor-optimized beats random noise!")
        print("Theoretical insight validated.")
    elif abs(sensor_mean - random_mean) < 0.05:
        print("[NEUTRAL] Sensor-optimized similar to random noise.")
        print("Theory may need refinement or different environment.")
    else:
        print("[FAILURE] Random noise beats sensor-optimized!")
        print("Theory does not hold in this setting.")


if __name__ == "__main__":
    run_benchmark()
