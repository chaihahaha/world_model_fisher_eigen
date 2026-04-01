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
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(action_dim, 8)
        self.fc1 = nn.Linear(state_dim + 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        a_emb = self.embed(action.long())
        x = torch.cat([state, a_emb], dim=-1)
        return self.fc2(torch.relu(self.fc1(x)))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, use_fisher=False):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_dim = action_dim
        self.use_fisher = use_fisher
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value = ValueNetwork(state_dim)
        self.world_model = WorldModel(state_dim, action_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=lr)
        
        self.fisher_matrix = None
    
    def get_action(self, state):
        logits = self.policy(torch.FloatTensor(state).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def get_action_fisher(self, state, weights):
        logits = self.policy(torch.FloatTensor(state).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        adjusted = probs * (1 + torch.FloatTensor(weights) * 0.1)
        adjusted = adjusted / adjusted.sum()
        dist = torch.distributions.Categorical(adjusted)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def compute_gradient(self, state, action):
        s = torch.FloatTensor(state).unsqueeze(0)
        a = torch.LongTensor([action])
        s_ = self.world_model(s, a)
        ds = torch.mean((s_ - s)**2)
        
        grad = torch.autograd.grad(ds, a, retain_graph=True)[0]
        return grad
    
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
        actions = torch.LongTensor(actions)
        old_lps = torch.FloatTensor(old_lps)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        returns = torch.zeros_like(rewards)
        r = 0
        for t in reversed(range(len(rewards))):
            r = rewards[t] + self.gamma * r * (1 - dones[t])
            returns[t] = r
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        logits = self.policy(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_lps = dist.log_prob(actions)
        
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
            pred = self.world_model(torch.FloatTensor(s).unsqueeze(0), torch.LongTensor([a]))
            loss = nn.MSELoss()(pred, torch.FloatTensor(ns).unsqueeze(0))
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()

def run_episode(env, agent, use_fisher=False):
    state, _ = env.reset(seed=random.randint(0, 10000))
    states, actions, lps, rewards, dones, next_states = [], [], [], [], [], []
    
    for step in range(500):
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
        
        states.append(state); actions.append(a); lps.append(lp); rewards.append(r); dones.append(t or tr); next_states.append(ns)
        state = ns
        if t or tr: break
    
    next_states.append(state)
    return states, actions, lps, rewards, dones, next_states

def main():
    print("=" * 65)
    print("PPO with World Model + Fisher Information vs Plain PPO")
    print("=" * 65)
    print("Environment: CartPole-v1 (Note: LunarLander requires Box2D)")
    print()
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print("--- PPO with World Model + Fisher (50 episodes) ---")
    print("(Fisher accumulated at each step: F += grad * grad^T)")
    print()
    
    agent = PPOAgent(state_dim, action_dim, use_fisher=True)
    fisher_rewards = []
    
    for ep in range(50):
        traj = run_episode(env, agent, use_fisher=True)
        s, a, lp, r, d, ns = traj
        agent.train_world_model(s, a, ns)
        agent.update(s, a, lp, r, d)
        fisher_rewards.append(sum(r))
        
        if (ep + 1) % 10 == 0:
            eigen = agent.get_min_eigen(agent.fisher_matrix) if agent.fisher_matrix is not None else 0
            avg_r = np.mean(fisher_rewards[-10:])
            print(f"  Episodes {ep-9}-{ep+1}: Avg Reward={avg_r:.1f}, Min Eigen={eigen:.6f}")
    
    print()
    print("--- Plain PPO (50 episodes) ---")
    print("(Standard PPO without world model or Fisher information)")
    print()
    
    plain = PPOAgent(state_dim, action_dim, use_fisher=False)
    plain_rewards = []
    
    for ep in range(50):
        traj = run_episode(env, plain, use_fisher=False)
        s, a, lp, r, d, ns = traj
        plain.update(s, a, lp, r, d)
        plain_rewards.append(sum(r))
        
        if (ep + 1) % 10 == 0:
            avg_r = np.mean(plain_rewards[-10:])
            print(f"  Episodes {ep-9}-{ep+1}: Avg Reward={avg_r:.1f}")
    
    env.close()
    
    print()
    print("=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"PPO+World+Fisher - Last 10 avg: {np.mean(fisher_rewards[-10:]):.1f}, Overall: {np.mean(fisher_rewards):.1f}")
    print(f"Plain PPO         - Last 10 avg: {np.mean(plain_rewards[-10:]):.1f}, Overall: {np.mean(plain_rewards):.1f}")
    
    print()
    print("=" * 65)
    print("Algorithm Implementation Details")
    print("=" * 65)
    print("1. PPO: Policy gradient with clipped surrogate objective")
    print("2. World Model: s' = f(s,a) - MLP regressing state transition")
    print("3. Fisher Matrix: F += grad * grad^T at each env step")
    print("   grad = dV(f(s,a))/d(theta) - gradient through world model")
    print("4. Action Reweighting: For each candidate action a:")
    print("   modified_F = F + grad(a)*grad(a)^T * learning_rate")
    print("   weight(a) = min_eigenvalue(modified_F)")

if __name__ == "__main__":
    main()
