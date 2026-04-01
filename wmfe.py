import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_is_discrete=True):
        super().__init__()
        self.action_is_discrete = action_is_discrete
        
        if action_is_discrete:
            self.embed = nn.Embedding(action_dim, 8)
            action_in_dim = 8
        else:
            self.action_dim = action_dim
            action_in_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim + action_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if self.action_is_discrete:
            if action.dim() == 0:
                action = action.unsqueeze(0)
            a_emb = self.embed(action.long())
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)
            a_emb = action.float()
        
        x = torch.cat([state, a_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def compute_gradient(self, state, action):
        s = torch.FloatTensor(state).unsqueeze(0)
        
        if self.action_is_discrete:
            a = torch.LongTensor([action])
            a_emb = self.embed(a)
        else:
            a = torch.FloatTensor(action).unsqueeze(0)
            a_emb = a
        
        a_emb = a_emb.clone().detach().requires_grad_(True)
        
        x = torch.cat([s, a_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        ds = self.fc3(x) - s
        
        grad = torch.autograd.grad(ds, a_emb, grad_outputs=torch.ones_like(ds), retain_graph=True)[0]
        return grad.squeeze(0) if grad is not None else None

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, action_is_discrete=True):
        super().__init__()
        self.action_is_discrete = action_is_discrete
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        if action_is_discrete:
            self.head = nn.Linear(hidden_dim // 2, action_dim)
        else:
            self.mean = nn.Linear(hidden_dim // 2, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = self.net(state)
        if self.action_is_discrete:
            return self.head(x)
        else:
            return self.mean(x), self.log_std

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 use_fisher=False, action_is_discrete=True, hidden_dim=128):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_dim = action_dim
        self.use_fisher = use_fisher
        self.action_is_discrete = action_is_discrete
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_is_discrete)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.world_model = WorldModel(state_dim, action_dim, hidden_dim, action_is_discrete)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)
        
        self.fisher_matrix = None
    
    def get_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        
        if self.action_is_discrete:
            logits = self.policy(s)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()
        else:
            mean, log_std = self.policy(s)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)
            return action.cpu().numpy().squeeze(-1), dist.log_prob(action).sum(dim=-1).item()
    
    def get_action_fisher(self, state, weights):
        s = torch.FloatTensor(state).unsqueeze(0)
        
        if self.action_is_discrete:
            logits = self.policy(s)
            probs = torch.softmax(logits, dim=-1)
            adjusted = probs * (1 + torch.FloatTensor(weights) * 0.1)
            adjusted = adjusted / adjusted.sum()
            dist = torch.distributions.Categorical(adjusted)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()
        else:
            mean, log_std = self.policy(s)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.tanh(action)
            return action.cpu().numpy().squeeze(-1), dist.log_prob(action).sum(dim=-1).item()
    
    def compute_gradient(self, state, action):
        return self.world_model.compute_gradient(state, action)
    
    def get_min_eigen(self, matrix):
        if matrix is None:
            return 0.0
        try:
            eig = torch.linalg.eigvalsh(matrix)
            nonzero = eig[eig > 1e-5]
            return torch.min(nonzero).item() if len(nonzero) > 0 else 0.0
        except:
            return 0.0
    
    def compute_weights(self, state):
        if self.fisher_matrix is None:
            return None
        weights = []
        for a in range(self.action_dim):
            try:
                g = self.compute_gradient(state, a)
                if g is not None:
                    m = self.fisher_matrix + (g.view(-1,1) @ g.view(1,-1)) * 0.01
                    weights.append(self.get_min_eigen(m))
                else:
                    weights.append(0.0)
            except:
                weights.append(0.0)
        if weights and max(weights) > 0:
            w = np.array(weights)
            return (w - w.min() + 1e-6) / (w.max() - w.min() + 1e-6)
        return np.ones(self.action_dim)
    
    def update(self, states, actions, old_lps, rewards, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(actions) if not self.action_is_discrete else torch.LongTensor(actions)
        old_lps = torch.FloatTensor(old_lps)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        returns = torch.zeros_like(rewards)
        r = 0
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.gamma * r * (1 - dones[t])
            returns[t] = r
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        if self.action_is_discrete:
            logits = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(actions)
        else:
            means, log_stds = self.policy(states)
            stds = log_stds.exp()
            dist = torch.distributions.Normal(means, stds)
            new_lps = dist.log_prob(actions).sum(dim=-1)
        
        ratio = torch.exp(new_lps - old_lps)
        advantages = returns - self.value(states).squeeze()
        
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        value_loss = nn.MSELoss()(self.value(states).squeeze(), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def train_world_model(self, states, actions, next_states):
        for s, a, ns in zip(states[:-1], actions[:-1], next_states[1:]):
            s_t = torch.FloatTensor(s).unsqueeze(0)
            a_t = torch.FloatTensor(a).unsqueeze(0) if not self.action_is_discrete else torch.LongTensor([a])
            ns_t = torch.FloatTensor(ns).unsqueeze(0)
            
            pred = self.world_model(s_t, a_t)
            loss = nn.MSELoss()(pred, ns_t)
            
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()

def run_episode(env, agent, use_fisher=False, max_steps=500):
    state, _ = env.reset(seed=random.randint(0, 10000))
    states, actions, lps, rewards, dones, next_states = [], [], [], [], [], []
    
    for step in range(max_steps):
        if use_fisher and agent.fisher_matrix is not None:
            w = agent.compute_weights(state)
            a, lp = agent.get_action_fisher(state, w)
        else:
            a, lp = agent.get_action(state)
        
        ns, r, t, tr, _ = env.step(a)
        
        if use_fisher:
            g = agent.compute_gradient(state, a)
            if g is not None:
                grad_outer = g.view(-1,1) @ g.view(1,-1)
                if agent.fisher_matrix is None:
                    agent.fisher_matrix = grad_outer + torch.eye(grad_outer.shape[0]) * 1e-6
                else:
                    agent.fisher_matrix = agent.fisher_matrix + grad_outer
        
        states.append(state)
        actions.append(a)
        lps.append(lp)
        rewards.append(r)
        dones.append(t or tr)
        next_states.append(ns)
        state = ns
        if t or tr: break
    
    next_states.append(state)
    return states, actions, lps, rewards, dones, next_states

def run_benchmark(env_name, n_episodes=200, max_steps=500, print_every=20):
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print(f"{'='*60}")
    
    env = gym.make(env_name)
    
    obs_space = env.observation_space
    act_space = env.action_space
    
    if isinstance(act_space, gym.spaces.Discrete):
        state_dim = obs_space.shape[0]
        action_dim = act_space.n
        action_is_discrete = True
    elif isinstance(act_space, gym.spaces.Box):
        state_dim = obs_space.shape[0]
        action_dim = act_space.shape[0]
        action_is_discrete = False
    else:
        print("Unsupported action space")
        return None, None
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Discrete: {action_is_discrete}")
    
    agent_fisher = PPOAgent(state_dim, action_dim, use_fisher=True, action_is_discrete=action_is_discrete, hidden_dim=128)
    fisher_rewards = []
    
    for ep in range(n_episodes):
        traj = run_episode(env, agent_fisher, use_fisher=True, max_steps=max_steps)
        s, a, lp, r, d, ns = traj
        agent_fisher.train_world_model(s, a, ns)
        agent_fisher.update(s, a, lp, r, d)
        fisher_rewards.append(sum(r))
        
        if (ep + 1) % print_every == 0:
            eigen = agent_fisher.get_min_eigen(agent_fisher.fisher_matrix) if agent_fisher.fisher_matrix is not None else 0
            avg_r = np.mean(fisher_rewards[-print_every:])
            print(f"  Fisher - Episodes {ep-print_every+2}-{ep+1}: Avg Reward={avg_r:.1f}, Min Eigen={eigen:.6f}")
    
    print(f"\n  Fisher final 10 episodes avg: {np.mean(fisher_rewards[-10:]):.1f}")
    
    agent_plain = PPOAgent(state_dim, action_dim, use_fisher=False, action_is_discrete=action_is_discrete, hidden_dim=128)
    plain_rewards = []
    
    for ep in range(n_episodes):
        traj = run_episode(env, agent_plain, use_fisher=False, max_steps=max_steps)
        s, a, lp, r, d, ns = traj
        agent_plain.update(s, a, lp, r, d)
        plain_rewards.append(sum(r))
        
        if (ep + 1) % print_every == 0:
            avg_r = np.mean(plain_rewards[-print_every:])
            print(f"  Plain  - Episodes {ep-print_every+2}-{ep+1}: Avg Reward={avg_r:.1f}")
    
    print(f"  Plain final 10 episodes avg: {np.mean(plain_rewards[-10:]):.1f}")
    
    env.close()
    
    return fisher_rewards, plain_rewards

def main():
    print("=" * 65)
    print("PPO with World Model + Fisher Information vs Plain PPO")
    print("Benchmark on Multiple Gymnasium Environments")
    print("=" * 65)
    
    benchmarks = [
        ("CartPole-v1", 100, 500),
        ("Pendulum-v1", 100, 500),
        ("MountainCar-v0", 100, 500),
        ("Acrobot-v1", 100, 500),
    ]
    
    results = {}
    for env_name, n_eps, max_steps in benchmarks:
        fisher_r, plain_r = run_benchmark(env_name, n_eps, max_steps)
        if fisher_r is not None:
            results[env_name] = (fisher_r, plain_r)
    
    print("\n" + "=" * 65)
    print("Benchmark Results Summary")
    print("=" * 65)
    print(f"{'Environment':<25} {'Fisher (last 10)':<18} {'Plain (last 10)':<18}")
    print("-" * 65)
    for env_name, (fisher_r, plain_r) in results.items():
        print(f"{env_name:<25} {np.mean(fisher_r[-10:]):<18.1f} {np.mean(plain_r[-10:]):<18.1f}")

if __name__ == "__main__":
    main()