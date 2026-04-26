"""
Action-particle Monte Carlo exploration for MC-PILCO.

CascadedGPExploration: Evaluates candidate actions for the current timestep
by predicting next-state Gaussians via GP dynamics, then running particle-based
rollouts from the predicted next states.

Algorithm:
  1. Sample N candidate actions for the current timestep
     (around control policy output + exploration noise)
  2. For each candidate action a_i:
     a. Predict next-state Gaussian from (current_state, a_i) via GP
     b. From that next state, run rollout using control policy + action particles:
        - Get control policy action for current state
        - Sample K action particles around control policy action
        - For each action particle, predict next-state Gaussian via GP
        - Merge K next-state Gaussians into one Gaussian
        - Use merged Gaussian as recursive input for next step
     c. Score = accumulated state variance (exploration coverage)
  3. Return the candidate action with highest coverage

Key insight: particles are for ACTIONS (not states). The GP accepts Gaussian
state inputs, so we merge action-particle predictions into a single Gaussian
for recursive rollout.
"""

import numpy as np
import torch

from policy_learning.Policy import Policy


class CascadedGPExploration(Policy):
    """
    Exploration policy using action-particle Monte Carlo sampling on the GP
    dynamics model with Gaussian merging at each timestep.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        f_control_policy=None,
        control_policy_par=None,
        flg_squash=True,
        u_max=1.0,
        num_particles=20,
        rollout_horizon=4,
        num_candidates=10,
        action_noise_std=2.0,
        Thompson_alpha=0.5,
        random_ratio=0.1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(CascadedGPExploration, self).__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            flg_squash=flg_squash,
            u_max=u_max,
            dtype=dtype,
            device=device,
        )
        self.u_max = u_max
        self.num_particles = num_particles
        self.rollout_horizon = rollout_horizon
        self.num_candidates = num_candidates
        self.action_noise_std = action_noise_std
        self.Thompson_alpha = Thompson_alpha
        self.random_ratio = random_ratio
        self.rng = np.random.RandomState(None)

        self.model_learning = None
        self.norm_list = None
        self.control_policy = None

        # Store control policy factory for lazy initialization
        self.f_control_policy = f_control_policy
        self.control_policy_par = control_policy_par

    def set_model_learning(self, model_learning):
        self.model_learning = model_learning
        self.norm_list = model_learning.norm_list

        # Initialize control policy if factory provided
        if self.f_control_policy is not None and self.control_policy_par is not None:
            self.control_policy = self.f_control_policy(**self.control_policy_par)

    def _random_action(self):
        rand_u = self.u_max * (2 * self.rng.rand(self.input_dim) - 1)
        if self.input_dim == 1:
            return torch.tensor(rand_u, dtype=self.dtype, device=self.device)
        return torch.tensor(
            rand_u.reshape([-1, self.input_dim]),
            dtype=self.dtype,
            device=self.device,
        )

    def _predict_next_state_gaussian(self, state_mean, state_var, action):
        """
        Given state Gaussian and a deterministic action, predict next-state
        Gaussian via the GP dynamics model.

        Returns (next_mean, next_var) as 1D tensors.
        """
        state_batch = state_mean.unsqueeze(0)
        action_batch = action.unsqueeze(0)

        gp_inputs, _, gp_mean_list, gp_var_list = self.model_learning.get_one_step_gp_out(
            states=state_batch, inputs=action_batch
        )

        # Apply normalization scaling to GP variance
        for i in range(len(gp_var_list)):
            gp_var_list[i] = gp_var_list[i] * (self.norm_list[i] ** 2)

        # Concatenate GP outputs
        delta_mean = torch.cat(gp_mean_list, dim=1).squeeze(0)
        delta_var = torch.cat(gp_var_list, dim=1).squeeze(0)

        # Compute next state mean and variance through dynamics
        next_mean = torch.zeros_like(state_mean)
        next_var = torch.zeros_like(state_mean)

        if hasattr(self.model_learning, 'vel_indeces'):
            # Speed model: GP predicts delta velocity
            next_mean[self.model_learning.vel_indeces] = (
                state_mean[self.model_learning.vel_indeces] + delta_mean
            )
            next_var[self.model_learning.vel_indeces] = (
                state_var[self.model_learning.vel_indeces] + delta_var
            )
            next_mean[self.model_learning.not_vel_indeces] = (
                state_mean[self.model_learning.not_vel_indeces]
                + self.model_learning.T_sampling * state_mean[self.model_learning.vel_indeces]
                + self.model_learning.T_sampling / 2 * delta_mean
            )
            next_var[self.model_learning.not_vel_indeces] = (
                state_var[self.model_learning.not_vel_indeces]
                + (self.model_learning.T_sampling ** 2) * state_var[self.model_learning.vel_indeces]
                + (self.model_learning.T_sampling ** 2 / 4) * delta_var
            )
        else:
            # Direct model: GP predicts state delta
            next_mean = state_mean + delta_mean
            next_var = state_var + delta_var

        return next_mean, next_var

    def _merge_gaussians(self, means_list, vars_list, weights=None):
        """
        Merge K Gaussians into one using moment matching.
        """
        K = len(means_list)
        if weights is None:
            weights = torch.ones(K, dtype=self.dtype, device=self.device) / K

        means_stacked = torch.stack(means_list)
        vars_stacked = torch.stack(vars_list)

        weights_col = weights.view(-1, 1)
        merged_mean = (weights_col * means_stacked).sum(dim=0)

        mean_diff_sq = (means_stacked - merged_mean.unsqueeze(0)) ** 2
        merged_var = (weights_col * (vars_stacked + mean_diff_sq)).sum(dim=0)
        merged_var = torch.clamp(merged_var, min=1e-8)

        return merged_mean, merged_var

    def _rollout_coverage(self, state_mean, state_var):
        """
        Run particle-based rollout from given state Gaussian using control
        policy + action particles. Returns accumulated state variance score.
        """
        accumulated_coverage = 0.0

        for step in range(self.rollout_horizon):
            # Get control policy action for current state
            if self.control_policy is not None:
                state_batch = state_mean.unsqueeze(0)
                ctrl_action = self.control_policy.forward(state_batch, t=step)
                ctrl_action_np = ctrl_action.detach().cpu().numpy().flatten()
            else:
                ctrl_action_np = self.rng.uniform(-self.u_max, self.u_max, self.input_dim)

            # Sample action particles around control policy action
            means_list = []
            vars_list = []
            for _ in range(self.num_particles):
                noise = self.rng.randn(self.input_dim) * self.action_noise_std * 0.5
                a_np = np.clip(ctrl_action_np + noise, -self.u_max, self.u_max)
                a = torch.tensor(a_np, dtype=self.dtype, device=self.device)
                next_mean, next_var = self._predict_next_state_gaussian(
                    state_mean, state_var, a
                )
                means_list.append(next_mean)
                vars_list.append(next_var)

            # Merge all action-particle predictions into one Gaussian
            merged_mean, merged_var = self._merge_gaussians(means_list, vars_list)

            # Thompson sampling: sample from posterior for exploration
            if self.Thompson_alpha > 0:
                noise = torch.randn_like(merged_mean) * torch.sqrt(
                    torch.clamp(merged_var, min=1e-8)
                ) * self.Thompson_alpha
                state_mean = merged_mean + noise
            else:
                state_mean = merged_mean
            state_var = merged_var

            # Track exploration coverage
            accumulated_coverage += state_var.sum().item()

        return accumulated_coverage

    def forward(self, states, t):
        # Fallback to random if no model or no training data
        if self.model_learning is None:
            return self._random_action()
        if hasattr(self.model_learning, "num_samples") and self.model_learning.num_samples == 0:
            return self._random_action()

        # Random exploration with some probability
        if self.rng.rand() < self.random_ratio:
            return self._random_action()

        # Extract current state
        if states.dim() == 2:
            current_state = states[0, :]
        else:
            current_state = states.clone()

        # Small initial state uncertainty
        state_var = torch.ones_like(current_state) * 1e-4

        # Score each candidate action:
        # 1. Predict next-state Gaussian from (current_state, candidate_action)
        # 2. Rollout from predicted next state using control policy + action particles
        scores = []
        candidate_actions = []

        for _ in range(self.num_candidates):
            # Sample candidate action around control policy output + noise
            if self.control_policy is not None:
                state_batch = current_state.unsqueeze(0)
                ctrl_action = self.control_policy.forward(state_batch, t=0)
                ctrl_action_np = ctrl_action.detach().cpu().numpy().flatten()
            else:
                ctrl_action_np = self.rng.uniform(-self.u_max, self.u_max, self.input_dim)

            noisy_action = ctrl_action_np + self.rng.randn(self.input_dim) * self.action_noise_std
            noisy_action = np.clip(noisy_action, -self.u_max, self.u_max)
            cand_action = torch.tensor(noisy_action, dtype=self.dtype, device=self.device)
            candidate_actions.append(cand_action)

            try:
                # Predict next-state Gaussian from candidate action
                next_mean, next_var = self._predict_next_state_gaussian(
                    current_state, state_var, cand_action
                )

                # Score by rolling out from the predicted next state
                coverage = self._rollout_coverage(next_mean, next_var)
                scores.append(-coverage)  # Negate since argmin selects best
            except Exception:
                scores.append(0.0)

        # Select best candidate action (highest coverage)
        best_idx = np.argmin(scores)
        best_action = candidate_actions[best_idx]

        out = best_action.detach().cpu().numpy()
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(axis=-1)
        return out

    def get_np_policy(self):
        return self.forward

    def record_step(self, state, action, next_state):
        pass

    def reset(self):
        pass

    def reinit(self, scaling=1):
        self.reset()
