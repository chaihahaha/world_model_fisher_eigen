import torch
import torch.nn as nn
import gpytorch
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

# --- 1. Dynamics Model (GP) ---
class DynamicsGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. RBF Policy ---
class RBFPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_centers=20):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, state_dim))
        self.weights = nn.Parameter(torch.randn(num_centers, action_dim))
        self.log_lengthscale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        dist = torch.cdist(x, self.centers)
        rbf = torch.exp(-0.5 * (dist / self.log_lengthscale.exp())**2)
        return torch.tanh(rbf @ self.weights)

# --- 3. PILCO-SensorOpt Engine ---
class PILCOSensorOpt:
    def __init__(self, state_dim, action_dim, horizon=25):
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = RBFPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.lam = 0.05 # Exploration weight
        
        self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(state_dim)]
        self.models = [None] * state_dim

    def update_dynamics(self, X, Y):
        """ X: [N, D+F], Y: [N, D] """
        print("Training Dynamics GPs...")
        for i in range(self.state_dim):
            self.models[i] = DynamicsGP(X, Y[:, i], self.likelihoods[i])
            self.models[i].train()
            self.likelihoods[i].train()

            # Basic Hyperparameter Optimization
            optimizer = torch.optim.Adam(self.models[i].parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[i], self.models[i])

            for _ in range(50): # Optimization steps
                optimizer.zero_grad()
                output = self.models[i](X)
                loss = -mll(output, Y[:, i])
                loss.backward()
                optimizer.step()
            
            # CRITICAL: Switch to eval mode for inference in the policy loop
            self.models[i].eval()
            self.likelihoods[i].eval()

    def cost_fn(self, state):
        target = torch.tensor([0.45, 0.0])
        dist = torch.sum((state - target)**2, dim=-1)
        return 1.0 - torch.exp(-0.5 * dist / 0.1)

    def optimize_policy(self, start_mu):
        self.optimizer.zero_grad()
        
        mu = start_mu
        total_task_loss = 0
        exploration_reward = 0
        
        # Analytic Propagation
        for t in range(self.horizon):
            action = self.policy(mu.unsqueeze(0)).squeeze(0)
            inp = torch.cat([mu, action]).unsqueeze(0)
            
            next_mu_diffs = []
            variances = []
            
            # Predict each dimension
            for i in range(self.state_dim):
                # We need gradients through the GP mean/var for policy optimization
                # Evaluation mode + Autograd = Analytic Gradients
                pred = self.models[i](inp)
                next_mu_diffs.append(pred.mean)
                variances.append(pred.variance)
            
            delta_mu = torch.cat(next_mu_diffs)
            sigma_f = torch.cat(variances)
            
            # Update state distribution mean
            mu = mu + delta_mu
            
            # 1. Task Cost (Minimize)
            total_task_loss += self.cost_fn(mu)
            
            # 2. SensorOpt Exploration (Maximize uncertainty in high-sensitivity regions)
            # In this simple version, we reward high model variance
            exploration_reward += torch.sum(sigma_f)
            
        # Loss = Task Cost - Lambda * Information Gain
        loss = total_task_loss - self.lam * exploration_reward
        loss.backward()
        self.optimizer.step()
        return total_task_loss.item()

# --- 4. Benchmark Logic ---
def run_benchmark():
    env = gym.make("MountainCarContinuous-v0")
    
    print("Collecting Initial Data...")
    obs, _ = env.reset()
    X, Y = [], []
    for _ in range(2000):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        X.append(np.concatenate([obs.flatten(), action.flatten()]))
        Y.append(next_obs - obs)
        obs = next_obs if not done else env.reset()[0]
    
    X_tensor = torch.tensor(np.array(X)).float()
    Y_tensor = torch.tensor(np.array(Y)).float()
    
    pilco = PILCOSensorOpt(state_dim=2, action_dim=1, horizon=30)
    pilco.update_dynamics(X_tensor, Y_tensor)
    
    print("Optimizing Policy...")
    history = []
    for i in range(100):
        # Start from the bottom of the valley
        start_state = torch.tensor([-0.5, 0.0]) 
        loss = pilco.optimize_policy(start_state)
        history.append(loss)
        if i % 10 == 0:
            print(f"Iter {i}, Task Cost: {loss:.4f}")

    # Plotting results
    plt.plot(history)
    plt.xlabel("Policy Iterations")
    plt.ylabel("Expected Task Cost")
    plt.title("PILCO-SensorOpt Convergence")
    plt.savefig('result.png')

if __name__ == "__main__":
    run_benchmark()
