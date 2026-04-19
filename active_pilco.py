import torch
import torch.nn as nn
import gpytorch
import gymnasium as gym
import numpy as np

# --- 1. Dynamics (GP) ---
class DynamicsGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

# --- 2. Policy ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# --- 3. PILCO Agent ---
class PILCOAgent:
    def __init__(self, state_dim=2, action_dim=1):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.models = [None] * state_dim
        self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(state_dim)]
        self.horizon = 45
        self.target = torch.tensor([0.83, 0.0]) # Normalized goal position
        # HIGHER W: This forces the agent to reduce sigma to get any reward
        self.W = torch.diag(torch.tensor([50.0, 5.0])) 

    def train_dynamics(self, S, A, NS):
        X = torch.cat([S, A], dim=-1)
        Y = NS - S
        for i in range(2):
            self.likelihoods[i].noise = 1e-4
            self.models[i] = DynamicsGP(X, Y[:, i], self.likelihoods[i])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[i], self.models[i])
            opt = torch.optim.Adam(self.models[i].parameters(), lr=0.03)
            self.models[i].train()
            for _ in range(60):
                opt.zero_grad()
                loss = -mll(self.models[i](X), Y[:, i])
                loss.backward()
                opt.step()
            self.models[i].eval()

    def expected_cost(self, mu, sigma):
        # Strict Analytic Expected Saturated Cost (Eq 24-25)
        mu_diff = mu - self.target
        # Sensor Logic: det term must be large enough to penalize sigma
        det_term = torch.det(torch.eye(2) + sigma @ self.W)
        S_inv = torch.inverse(sigma + torch.inverse(self.W))
        # Note: reward = exp(...) / sqrt(det)
        reward = torch.exp(-0.5 * mu_diff @ S_inv @ mu_diff.t()) / torch.sqrt(det_term + 1e-8)
        return 1.0 - reward

    def optimize_policy(self):
        self.optimizer.zero_grad()
        mu = torch.tensor([[-0.22, 0.0]], requires_grad=True) # Normalized bottom
        sigma = torch.eye(2).unsqueeze(0) * 1e-3
        
        total_j = 0
        total_u2 = 0

        for t in range(self.horizon):
            u = self.policy(mu)
            inp = torch.cat([mu, u], dim=-1)
            
            d_mu_list, d_var_list = [], []
            for i in range(2):
                dist = self.models[i](inp)
                d_mu_list.append(dist.mean)
                d_var_list.append(dist.variance.clamp(min=1e-6))
            
            mu_delta = torch.stack(d_mu_list, dim=-1).view(1, 2)
            var_delta = torch.diag_embed(torch.stack(d_var_list, dim=-1)).view(1, 2, 2)
            
            # --- Eq 11-12: Full Covariance Update ---
            # Using Taylor expansion to include cross-covariance (Momentum)
            # This makes the agent realize uncertainty builds up.
            mu_prev = mu.clone()
            mu = mu + mu_delta
            
            # Cross-covariance approximation: cov(x, f(x)) = sigma @ grad(f)^T
            # For simplicity and stability, we use a compounding growth term
            sigma = sigma + var_delta + 0.1 * (sigma @ var_delta + var_delta @ sigma)

            # Task Objective (Task + Energy Penalty)
            cost = self.expected_cost(mu, sigma)
            u_penalty = 0.05 * (u**2).sum() # Standard RL action regularization
            total_j += (cost + u_penalty)
            total_u2 += (u**2).sum()

        total_j.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        vol = torch.logdet(sigma[0]).item()
        return total_j.item(), mu[0, 0].item(), vol, (total_u2/self.horizon).item()

# --- 4. Main ---
def run():
    env = gym.make("MountainCarContinuous-v0")
    agent = PILCOAgent()
    def norm(s): return np.array([(s[0] + 0.3)/0.9, s[1]/0.07])

    S_b, A_b, NS_b = [], [], []
    obs, _ = env.reset()
    for _ in range(500):
        act = env.action_space.sample()
        n_obs, _, te, tr, _ = env.step(act)
        S_b.append(norm(obs)); A_b.append(act); NS_b.append(norm(n_obs))
        obs = n_obs if not (te or tr) else env.reset()[0]

    for trial in range(12):
        agent.train_dynamics(torch.tensor(np.array(S_b)).float(),
                            torch.tensor(np.array(A_b)).float(),
                            torch.tensor(np.array(NS_b)).float())
        
        print(f"\n--- Trial {trial} Optimization ---")
        for step in range(41):
            loss, pos, vol, u2 = agent.optimize_policy()
            if step % 20 == 0:
                print(f"  Step {step:2d} | Hallucinated Cost: {loss:5.2f} | Pred Pos: {pos:5.2f} | LogInfoVol: {vol:7.3f} | AvgU2: {u2:4.2f}")

        obs, _ = env.reset()
        ret, max_p = 0, -1.2
        for _ in range(200):
            with torch.no_grad():
                act = agent.policy(torch.tensor(norm(obs)).float().unsqueeze(0)).numpy()[0]
            n_obs, r, te, tr, _ = env.step(act)
            S_b.append(norm(obs)); A_b.append(act); NS_b.append(norm(n_obs))
            ret += r; obs = n_obs; max_p = max(max_p, obs[0])
            if te or tr: break
            
        print(f"Trial {trial} RESULT | Return: {ret:7.2f} | Max Pos: {max_p:5.2f} | Samples: {len(S_b)}")
        if ret > 90: break

if __name__ == "__main__":
    run()
