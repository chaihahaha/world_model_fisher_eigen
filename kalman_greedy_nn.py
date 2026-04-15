"""
Kalman-Greedy Neural Network (KGNN)

NEURAL NETWORK VERSION OF KALMAN-GREEDY:

Original Kalman-Greedy uses Riccati equation for uncertainty:
  IG(s) = φ(s)^T P² φ(s) / (φ(s)^T P φ(s) + σ²)

Neural Network Version uses MC Dropout as Bayesian approximation:
  Uncertainty(s) = Var_{dropout}[f_θ(s)]

KEY CHANGES:
1. Random features → Neural network feature extractor
2. Covariance matrix P → MC Dropout prediction variance
3. Tabular Q → Neural network Q-function
4. Tabular transition model → Optional world model

ARCHITECTURE:
  State s → [Feature Net] → z → [Value Head] → V(s)
                            ↓
                     [MC Dropout]
                            ↓
                     Uncertainty estimate

EXPLORATION BONUS:
  UCB(s,a) = Q(s,a) + β × sqrt(Var[Q(s,a)])

This is equivalent to Kalman-Greedy with:
  - Neural network replaces φ(s)
  - MC Dropout variance replaces φ(s)^T P² φ(s)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


class KalmanGreedyNet(nn.Module):
    """Neural network with MC Dropout for uncertainty estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout_p: float = 0.5, n_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout_p = dropout_p
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_with_uncertainty(self, state: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self(state))
        predictions = torch.stack(predictions, dim=0)
        return predictions.mean(dim=0), predictions.var(dim=0)


class KalmanGreedyNetContinuous(KalmanGreedyNet):
    """For continuous action spaces. Outputs: mean_action, action_std, value"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout_p: float = 0.5, action_scale: float = 1.0):
        super().__init__(state_dim, action_dim, hidden_dim, dropout_p)
        self.action_mean_head = nn.Linear(hidden_dim, action_dim)
        self.action_logstd_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.action_scale = action_scale
        del self.network[-1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.network(x)
        action_mean = torch.tanh(self.action_mean_head(features)) * self.action_scale
        action_logstd = self.action_logstd_head(features)
        action_std = torch.exp(action_logstd).clamp(0.1, 2.0)
        value = self.value_head(features)
        return action_mean, action_std, value
    
    def predict_with_uncertainty(self, state: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        action_means = []
        with torch.no_grad():
            for _ in range(n_samples):
                am, _, _ = self(state)
                action_means.append(am)
        action_means = torch.stack(action_means, dim=0)
        return action_means.mean(dim=0), action_means.var(dim=0)


class KalmanGreedyAgentNN:
    """Kalman-Greedy with Neural Networks and MC Dropout for discrete actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout_p: float = 0.5,
                 lr: float = 1e-3, gamma: float = 0.99, beta: float = 1.0, mc_samples: int = 10,
                 buffer_size: int = 100000, batch_size: int = 64, target_update: int = 100, device: str = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta
        self.mc_samples = mc_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        self.q_network = KalmanGreedyNet(state_dim, action_dim, hidden_dim, dropout_p).to(self.device)
        self.target_network = KalmanGreedyNet(state_dim, action_dim, hidden_dim, dropout_p).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        self.episodes = []
        self.episode_reward = 0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(q_values.argmax().item())
        
        q_mean, q_variance = self.q_network.predict_with_uncertainty(state_tensor, self.mc_samples)
        uncertainty_bonus = self.beta * torch.sqrt(q_variance)
        scores = q_mean + uncertainty_bonus
        return int(scores.argmax().item())
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        self.target_network.eval()
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = nn.MSELoss()(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.store_transition(state, action, reward, next_state, done)
        if len(self.buffer) >= self.batch_size:
            self.update()
        self.episode_reward += reward
        self.total_steps += 1
        if done:
            self.episodes.append(self.episode_reward)
            self.episode_reward = 0
    
    def get_stats(self) -> Dict:
        return {"episodes": len(self.episodes), "recent_reward": float(np.mean(self.episodes[-10:])) if self.episodes else 0.0, "total_steps": self.total_steps}


class KalmanGreedyAgentNNContinuous:
    """Kalman-Greedy for continuous action spaces with actor-critic architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dropout_p: float = 0.5,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, gamma: float = 0.99, beta: float = 1.0,
                 mc_samples: int = 10, buffer_size: int = 100000, batch_size: int = 64, device: str = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta
        self.mc_samples = mc_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        self.actor = KalmanGreedyNetContinuous(state_dim, action_dim, hidden_dim, dropout_p).to(self.device)
        self.critic = KalmanGreedyNet(state_dim, action_dim, hidden_dim, dropout_p).to(self.device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episodes = []
        self.episode_reward = 0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            self.actor.eval()
            with torch.no_grad():
                action_mean, action_std, _ = self.actor(state_tensor)
                action = action_mean.cpu().numpy()[0]
            return action
        
        action_mean, action_std, _ = self.actor(state_tensor)
        _, uncertainty_var = self.actor.predict_with_uncertainty(state_tensor, self.mc_samples)
        action_std_total = action_std + self.beta * torch.sqrt(uncertainty_var.mean(dim=1, keepdim=True))
        
        self.actor.train()
        epsilon = torch.randn_like(action_mean)
        action = (action_mean + action_std_total * epsilon).clamp(-1, 1)
        return action.detach().cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def update(self) -> Tuple[float, float]:
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0
        
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Actor forward for current states
        action_mean, action_std, _ = self.actor(states)
        
        # Get target value from next states (no grad)
        self.actor.eval()
        with torch.no_grad():
            _, _, next_value = self.actor(next_states)
            target_value = (rewards + self.gamma * next_value * (1 - dones)).detach()
        
        self.actor.train()
        
        # Policy gradient update
        log_prob = -0.5 * torch.sum(torch.log(2 * np.pi * action_std**2) + (actions - action_mean)**2 / action_std**2, dim=1)
        policy_loss = -(log_prob * target_value.squeeze(1)).mean()
        
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()
        
        return 0.0, policy_loss.item()
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.store_transition(state, action, reward, next_state, done)
        if len(self.buffer) >= self.batch_size:
            self.update()
        self.episode_reward += reward
        self.total_steps += 1
        if done:
            self.episodes.append(self.episode_reward)
            self.episode_reward = 0
    
    def get_stats(self) -> Dict:
        return {"episodes": len(self.episodes), "recent_reward": float(np.mean(self.episodes[-10:])) if self.episodes else 0.0, "total_steps": self.total_steps}
