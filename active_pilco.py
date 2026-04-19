import torch
import torch.nn as nn
import gpytorch
import gymnasium as gym
import numpy as np

# --- 1. Dynamics Model (GP) ---
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

# --- 3. PILCO Agent (Task-Objective Only) ---
class PILCOAgent:
    def __init__(self):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.models = [None, None]
        self.likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(2)]
        self.horizon = 40
        self.target = torch.tensor([0.45, 0.0])

    def train_dynamics(self, S, A, NS):
        X = torch.cat([S, A], dim=-1)
        Y = NS - S
        for i in range(2):
            self.likelihoods[i].noise = 1e-4
            self.models[i] = DynamicsGP(X, Y[:, i], self.likelihoods[i])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[i], self.models[i])
            opt = torch.optim.Adam(self.models[i].parameters(), lr=0.03)
            self.models[i].train()
            for _ in range(50):
                opt.zero_grad()
                loss = -mll(self.models[i](X), Y[:, i])
                loss.backward()
                opt.step()
            self.models[i].eval()

    def get_expected_cost(self, mu, sigma):
        """
        Analytic Expected Saturated Cost (Eq 24-25 in pilco.md)
        E[1 - exp(-0.5 * (x-z)^T W (x-z))]
        """
        # W defines the precision of the target (how small the goal is)
        W = torch.diag(torch.tensor([10.0, 1.0])) 
        # mu: [batch, 2], sigma: [batch, 2, 2]
        # This formula penalizes high variance (sigma) automatically
        diff = mu - self.target
        S = sigma + torch.inverse(W)
        det_term = torch.det(torch.eye(2) + sigma @ W)
        exp_term = torch.exp(-0.5 * torch.sum(diff * (diff @ torch.inverse(S)), dim=-1))
        return 1.0 - exp_term / torch.sqrt(det_term + 1e-6)

    def optimize_policy(self):
        self.optimizer.zero_grad()
        
        # Initial Distribution: mu_0, sigma_0
        mu = torch.tensor([[-0.5, 0.0]])
        sigma = torch.eye(2).unsqueeze(0) * 1e-4
        
        total_j = 0
        
        for t in range(self.horizon):
            # 1. Action (incorporating action penalty into the task objective)
            u = self.policy(mu)
            
            # 2. Predict next distribution (Simplified Moment Matching)
            inp = torch.cat([mu, u], dim=-1)
            
            mu_new_list = []
            var_new_list = []
            for i in range(2):
                dist = self.models[i](inp)
                mu_new_list.append(mu[:, i] + dist.mean)
                # PILCO First Principle: Sigma_t = Sigma_{t-1} + Var_f + Cov...
                # We use the simplified update for stability
                var_new_list.append(sigma[:, i, i] + dist.variance)
            
            mu = torch.stack(mu_new_list, dim=-1)
            sigma = torch.diag_embed(torch.stack(var_new_list, dim=-1))
            
            # 3. Task Objective: Minimize Expected Cost + Action Energy
            # Exploration is EMERGENT because high sigma increases expected_cost
            total_j += self.get_expected_cost(mu, sigma).mean() + 0.1 * (u**2).sum()

        total_j.backward()
        self.optimizer.step()
        return total_j.item(), mu[0, 0].item(), sigma[0].diagonal().sum().item()

# --- 4. Main ---
def run():
    env = gym.make("MountainCarContinuous-v0")
    agent = PILCOAgent()
    
    # Normalization (Standardizing states for GP)
    def norm(s): return np.array([(s[0] + 0.3)/0.9, s[1]/0.07])
    def denorm_act(a): return np.clip(a, -1, 1)

    S_b, A_b, NS_b = [], [], []
    obs, _ = env.reset()
    for _ in range(400):
        act = env.action_space.sample()
        n_obs, r, te, tr, _ = env.step(act)
        S_b.append(norm(obs)); A_b.append(act); NS_b.append(norm(n_obs))
        obs = n_obs if not (te or tr) else env.reset()[0]

    for trial in range(10):
        agent.train_dynamics(torch.tensor(np.array(S_b)).float(),
                            torch.tensor(np.array(A_b)).float(),
                            torch.tensor(np.array(NS_b)).float())
        
        print(f"\n--- Trial {trial} Optimization ---")
        for step in range(31):
            loss, pos, var = agent.optimize_policy()
            if step % 10 == 0:
                print(f"  Step {step:2d} | Task Obj: {loss:5.2f} | Pred Pos: {pos:5.2f} | Pred Var: {var:7.5f}")

        # Real Execution
        obs, _ = env.reset()
        ret, max_p = 0, -1.2
        for _ in range(200):
            with torch.no_grad():
                act = agent.policy(torch.tensor(norm(obs)).float().unsqueeze(0)).numpy()[0]
            n_obs, r, te, tr, _ = env.step(denorm_act(act))
            S_b.append(norm(obs)); A_b.append(act); NS_b.append(norm(n_obs))
            ret += r; obs = n_obs; max_p = max(max_p, obs[0])
            if te or tr: break
            
        print(f"Trial {trial} RESULT | Return: {ret:7.2f} | Max Pos: {max_p:5.2f} | Data: {len(S_b)}")
        if ret > 90: break

if __name__ == "__main__":
    run()
