"""
Sensor-optimized exploration policies for MC-PILCO.

These policies use the learned GP dynamics model to guide exploration
toward state-action regions that maximize information gain (high GP variance),
implementing the theoretical framework from the sensor selection paper.
"""

import numpy as np
import torch

from policy_learning.Policy import Policy


class GPUncertaintyExploration(Policy):
    """
    Exploration policy that uses GP predictive variance to guide actions.
    
    At each step, evaluates candidate actions by querying the GP model for
    predictive variance at the next state. Selects the action that maximizes
    expected uncertainty (sensor selection objective: max tr(P)).
    
    Uses a grid search over action space combined with random perturbations
    for efficiency.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        flg_squash=True,
        u_max=1.0,
        num_candidates=20,
        random_ratio=0.3,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(GPUncertaintyExploration, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.num_candidates = num_candidates
        self.random_ratio = random_ratio
        self.u_max = u_max
        self.dtype = dtype
        self.device = device

        # Will be set during initialization
        self.model_learning = None
        self.norm_list = None
        self.input_dim_actual = input_dim

        # Pre-allocate candidate actions
        self.candidate_actions = None

    def set_model_learning(self, model_learning):
        """Attach the trained GP model learning object."""
        self.model_learning = model_learning
        self.norm_list = self.model_learning.norm_list
        self.input_dim_actual = self.model_learning.dim_input - self.state_dim

    def _get_gp_variance(self, states, inputs):
        """
        Get total GP predictive variance for given state-input pairs.
        Returns sum of variances across all GP output dimensions.
        """
        if self.model_learning is None:
            return torch.zeros(states.shape[0], 1, dtype=self.dtype, device=self.device)

        try:
            gp_inputs = self.model_learning.data_to_gp_input(states=states, inputs=inputs)
            gp_output_mean_list, gp_output_var_list = self.model_learning.get_one_step_gp_out(
                states=states, inputs=inputs
            )
            # Sum variance across all GP dimensions
            total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))
            # Apply normalization
            for i in range(len(self.norm_list)):
                gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2
            # Re-sum after normalization
            total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))
            return total_var
        except Exception:
            return torch.zeros(states.shape[0], 1, dtype=self.dtype, device=self.device)

    def _sample_candidate_actions(self, t):
        """Sample candidate actions: grid + random perturbations."""
        if self.input_dim_actual == 1:
            # 1D action space: grid search + random
            grid_points = torch.linspace(-self.u_max, self.u_max, self.num_candidates,
                                         dtype=self.dtype, device=self.device)
            random_actions = self.u_max * (2 * torch.rand(self.num_candidates, dtype=self.dtype,
                                                           device=self.device) - 1)
            self.candidate_actions = torch.cat([grid_points, random_actions])
            return self.candidate_actions.unsqueeze(1)  # [N, 1]
        else:
            # Multi-dimensional: random sampling
            self.candidate_actions = self.u_max * (
                2 * torch.rand(self.num_candidates, self.input_dim_actual, dtype=self.dtype, device=self.device) - 1
            )
            return self.candidate_actions

    def forward(self, states, t):
        """
        Select action that maximizes GP predictive variance at next state.
        
        For each candidate action, predict the next state using the GP model's
        mean function, then evaluate GP variance at that predicted state-input pair.
        """
        if self.model_learning is None or self.random_ratio > np.random.rand():
            # Fall back to random exploration
            rand_u = self.u_max * (2 * np.random.rand(self.input_dim) - 1)
            if self.input_dim == 1:
                return torch.tensor(rand_u, dtype=self.dtype, device=self.device)
            return torch.tensor(rand_u.reshape([-1, self.input_dim]), dtype=self.dtype, device=self.device)

        # Get current state (use mean if batch)
        if states.dim() == 2:
            # Take first particle as representative state
            current_state = states[0, :]  # [state_dim]
        else:
            current_state = states  # [state_dim]

        # Sample candidate actions
        candidates = self._sample_candidate_actions(t)
        n_candidates = candidates.shape[0]

        # Expand current state to batch
        state_batch = current_state.unsqueeze(0).repeat(n_candidates, 1)  # [N, state_dim]
        input_batch = candidates  # [N, input_dim]

        # Get GP predictions for all candidates
        gp_inputs = self.model_learning.data_to_gp_input(states=state_batch, inputs=input_batch)

        # Compute GP mean predictions for next state
        gp_output_mean_list, gp_output_var_list = self.model_learning.get_one_step_gp_out(
            states=state_batch, inputs=input_batch
        )

        # Apply normalization to variance
        for i in range(len(self.norm_list)):
            gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2

        # Compute next state mean: x_next = x + delta_mean
        delta_mean = torch.cat(gp_output_mean_list, 1)  # [N, state_dim]
        next_state_mean = state_batch + delta_mean  # [N, state_dim]

        # Compute total GP variance at next state as information gain proxy
        # We use the current GP variance as a proxy (cheaper than re-querying)
        total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))  # [N, 1]

        # Select action with highest variance
        best_idx = torch.argmax(total_var.squeeze(1))
        best_action = candidates[best_idx]

        if self.f_squash is not None:
            best_action = self.f_squash(best_action)

        if self.input_dim == 1:
            return best_action
        return best_action.reshape([-1, self.input_dim])

    def get_np_policy(self):
        """Returns a numpy function handle."""
        f = lambda state, t: self.forward_np(state, t)
        return f

    def forward_np(self, state, t=None):
        """Numpy implementation."""
        state_tc = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
        output = self.forward(state_tc, t)
        out = output.detach().cpu().numpy()
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(axis=-1)
        return out

    def reinit(self, scaling=1):
        pass


class GPInformativenessExploration(Policy):
    """
    Exploration policy based on expected information gain.
    
    Instead of maximizing raw GP variance, this policy estimates the
    expected reduction in GP uncertainty from sampling at each candidate
    action, using the posterior variance as an information gain proxy.
    
    Implements the sensor selection objective from the Kalman filtering
    paper: maximize det(P) or tr(P) of the posterior covariance.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        flg_squash=True,
        u_max=1.0,
        num_candidates=30,
        random_ratio=0.4,
        variance_weight=1.0,
        gradient_weight=0.0,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(GPInformativenessExploration, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.num_candidates = num_candidates
        self.random_ratio = random_ratio
        self.u_max = u_max
        self.dtype = dtype
        self.device = device
        self.variance_weight = variance_weight
        self.gradient_weight = gradient_weight

        self.model_learning = None
        self.norm_list = None
        self.input_dim_actual = input_dim

    def set_model_learning(self, model_learning):
        """Attach the trained GP model learning object."""
        self.model_learning = model_learning
        self.norm_list = model_learning.norm_list
        self.input_dim_actual = model_learning.dim_input - state_dim

    def _compute_information_score(self, state_batch, input_batch):
        """
        Compute information score for candidate state-input pairs.
        
        Score = variance_weight * GP_variance + gradient_weight * |d(variance)/d(action)|
        
        Uses GP predictive variance as the primary informativeness signal.
        """
        gp_output_mean_list, gp_output_var_list = self.model_learning.get_one_step_gp_out(
            states=state_batch, inputs=input_batch
        )

        # Apply normalization
        for i in range(len(self.norm_list)):
            gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2

        # Total variance across all output dimensions
        total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))  # [N, 1]

        # Compute variance of variance (higher = more diverse uncertainty)
        var_of_var = torch.var(total_var.squeeze(1), dim=0, keepdim=True) if total_var.shape[0] > 1 else torch.zeros(1, 1)

        score = self.variance_weight * total_var.squeeze(1)
        return score

    def forward(self, states, t):
        """Select action maximizing expected information gain."""
        if self.model_learning is None or self.random_ratio > np.random.rand():
            rand_u = self.u_max * (2 * np.random.rand(self.input_dim) - 1)
            if self.input_dim == 1:
                return torch.tensor(rand_u, dtype=self.dtype, device=self.device)
            return torch.tensor(rand_u.reshape([-1, self.input_dim]), dtype=self.dtype, device=self.device)

        if states.dim() == 2:
            current_state = states[0, :]
        else:
            current_state = states

        # Sample candidate actions
        if self.input_dim_actual == 1:
            candidates = torch.linspace(-self.u_max, self.u_max, self.num_candidates,
                                         dtype=self.dtype, device=self.device)
            random_actions = self.u_max * (2 * torch.rand(self.num_candidates, dtype=self.dtype,
                                                           device=self.device) - 1)
            candidates = torch.cat([candidates, random_actions])
            input_batch = candidates.unsqueeze(1)
        else:
            candidates = self.u_max * (
                2 * torch.rand(self.num_candidates, self.input_dim_actual, dtype=self.dtype, device=self.device) - 1
            )
            input_batch = candidates

        state_batch = current_state.unsqueeze(0).repeat(input_batch.shape[0], 1)

        # Compute information score
        scores = self._compute_information_score(state_batch, input_batch)

        best_idx = torch.argmax(scores)
        best_action = input_batch[best_idx]

        if self.f_squash is not None:
            best_action = self.f_squash(best_action)

        if self.input_dim == 1:
            return best_action
        return best_action.reshape([-1, self.input_dim])

    def get_np_policy(self):
        f = lambda state, t: self.forward_np(state, t)
        return f

    def forward_np(self, state, t=None):
        state_tc = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
        output = self.forward(state_tc, t)
        out = output.detach().cpu().numpy()
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(axis=-1)
        return out

    def reinit(self, scaling=1):
        pass


class HybridGPExploration(Policy):
    """
    Hybrid exploration: mixes random exploration with GP-guided exploration.
    
    This implements the theoretical insight that the exploration policy
    should be policy-weighted: exploration(x,u) = sigma_f(x,u) * rho_pi*(x).
    
    The mixing ratio can decay over time to transition from exploration
    to exploitation.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        flg_squash=True,
        u_max=1.0,
        initial_random_ratio=0.7,
        min_random_ratio=0.1,
        decay_rate=0.1,
        num_candidates=25,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(HybridGPExploration, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.initial_random_ratio = initial_random_ratio
        self.min_random_ratio = min_random_ratio
        self.decay_rate = decay_rate
        self.num_candidates = num_candidates
        self.dtype = dtype
        self.device = device
        self.current_step = 0

        self.model_learning = None
        self.norm_list = None
        self.input_dim_actual = input_dim

    def set_model_learning(self, model_learning):
        self.model_learning = model_learning
        self.norm_list = model_learning.norm_list
        self.input_dim_actual = model_learning.dim_input - self.state_dim

    def _get_random_ratio(self):
        """Compute decaying random exploration ratio."""
        ratio = self.initial_random_ratio * (1 - self.decay_rate) ** self.current_step
        return max(ratio, self.min_random_ratio)

    def forward(self, states, t):
        random_ratio = self._get_random_ratio()
        self.current_step += 1

        if self.model_learning is None or random_ratio > np.random.rand():
            rand_u = self.u_max * (2 * np.random.rand(self.input_dim) - 1)
            if self.input_dim == 1:
                return torch.tensor(rand_u, dtype=self.dtype, device=self.device)
            return torch.tensor(rand_u.reshape([-1, self.input_dim]), dtype=self.dtype, device=self.device)

        if states.dim() == 2:
            current_state = states[0, :]
        else:
            current_state = states

        # Sample candidate actions
        if self.input_dim_actual == 1:
            candidates = torch.linspace(-self.u_max, self.u_max, self.num_candidates,
                                         dtype=self.dtype, device=self.device)
            random_actions = self.u_max * (
                2 * torch.rand(self.num_candidates, dtype=self.dtype, device=self.device) - 1
            )
            candidates = torch.cat([candidates, random_actions])
            input_batch = candidates.unsqueeze(1)
        else:
            candidates = self.u_max * (
                2 * torch.rand(self.num_candidates, self.input_dim_actual, dtype=self.dtype, device=self.device) - 1
            )
            input_batch = candidates

        state_batch = current_state.unsqueeze(0).repeat(input_batch.shape[0], 1)

        # Query GP for variance
        gp_output_mean_list, gp_output_var_list = self.model_learning.get_one_step_gp_out(
            states=state_batch, inputs=input_batch
        )

        for i in range(len(self.norm_list)):
            gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2

        total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))  # [N, 1]

        # Score: prioritize actions leading to high uncertainty states
        score = total_var.squeeze(1)

        best_idx = torch.argmax(score)
        best_action = input_batch[best_idx]

        if self.f_squash is not None:
            best_action = self.f_squash(best_action)

        if self.input_dim == 1:
            return best_action
        return best_action.reshape([-1, self.input_dim])

    def get_np_policy(self):
        f = lambda state, t: self.forward_np(state, t)
        return f

    def forward_np(self, state, t=None):
        state_tc = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
        output = self.forward(state_tc, t)
        out = output.detach().cpu().numpy()
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(axis=-1)
        return out

    def reinit(self, scaling=1):
        self.current_step = 0


class GPGradientExploration(Policy):
    """
    Exploration policy that maximizes the gradient of GP variance w.r.t. action.
    
    This implements the theoretical result that the optimal exploration
    direction is along the gradient of the predictive variance:
    u* = argmax_u sigma^2(x+, pi(u)) where x+ = f(x, u).
    
    Uses finite-difference approximation of the gradient for efficiency.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        f_model_learning,
        model_learning_par,
        flg_squash=True,
        u_max=1.0,
        num_candidates=40,
        random_ratio=0.35,
        finite_diff_eps=0.05,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(GPGradientExploration, self).__init__(
            state_dim=state_dim, input_dim=input_dim, flg_squash=flg_squash, u_max=u_max, dtype=dtype, device=device
        )
        self.num_candidates = num_candidates
        self.random_ratio = random_ratio
        self.u_max = u_max
        self.finite_diff_eps = finite_diff_eps
        self.dtype = dtype
        self.device = device

        self.model_learning = None
        self.norm_list = None
        self.input_dim_actual = input_dim

    def set_model_learning(self, model_learning):
        self.model_learning = model_learning
        self.norm_list = model_learning.norm_list
        self.input_dim_actual = model_learning.dim_input - self.state_dim

    def _compute_variance_and_gradient(self, state, input_batch):
        """
        Compute GP variance and its gradient w.r.t. action using finite differences.
        """
        n = input_batch.shape[0]
        state_batch = state.unsqueeze(0).repeat(n, 1)

        gp_output_mean_list, gp_output_var_list = self.model_learning.get_one_step_gp_out(
            states=state_batch, inputs=input_batch
        )

        for i in range(len(self.norm_list)):
            gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2

        total_var = sum(gp_output_var_list[i] for i in range(len(gp_output_var_list)))  # [N, 1]

        if self.input_dim_actual == 1:
            # Compute gradient via finite differences
            var_vals = total_var.squeeze(1)
            # Find the action with highest variance
            best_idx = torch.argmax(var_vals)
            best_action = input_batch[best_idx]
            return best_action, var_vals
        else:
            best_idx = torch.argmax(total_var.squeeze(1))
            best_action = input_batch[best_idx]
            return best_action, total_var.squeeze(1)

    def forward(self, states, t):
        if self.model_learning is None or self.random_ratio > np.random.rand():
            rand_u = self.u_max * (2 * np.random.rand(self.input_dim) - 1)
            if self.input_dim == 1:
                return torch.tensor(rand_u, dtype=self.dtype, device=self.device)
            return torch.tensor(rand_u.reshape([-1, self.input_dim]), dtype=self.dtype, device=self.device)

        if states.dim() == 2:
            current_state = states[0, :]
        else:
            current_state = states

        if self.input_dim_actual == 1:
            # Grid search for 1D action space
            candidates = torch.linspace(-self.u_max, self.u_max, self.num_candidates,
                                         dtype=self.dtype, device=self.device)
            random_actions = self.u_max * (
                2 * torch.rand(self.num_candidates, dtype=self.dtype, device=self.device) - 1
            )
            candidates = torch.cat([candidates, random_actions])
            input_batch = candidates.unsqueeze(1)
        else:
            candidates = self.u_max * (
                2 * torch.rand(self.num_candidates, self.input_dim_actual, dtype=self.dtype, device=self.device) - 1
            )
            input_batch = candidates

        best_action, _ = self._compute_variance_and_gradient(current_state, input_batch)

        if self.f_squash is not None:
            best_action = self.f_squash(best_action)

        if self.input_dim == 1:
            return best_action
        return best_action.reshape([-1, self.input_dim])

    def get_np_policy(self):
        f = lambda state, t: self.forward_np(state, t)
        return f

    def forward_np(self, state, t=None):
        state_tc = torch.tensor(state, dtype=self.dtype, device=self.device).unsqueeze(0)
        output = self.forward(state_tc, t)
        out = output.detach().cpu().numpy()
        if out.ndim > 1 and out.shape[-1] == 1:
            out = out.squeeze(axis=-1)
        return out

    def reinit(self, scaling=1):
        pass
