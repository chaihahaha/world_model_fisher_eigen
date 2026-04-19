"""
Information-Theoretic GP-PILCO (IGP-PILCO)
==========================================

Migrating optimal sensor placement theory to exploration policy design
in model-based reinforcement learning.

The core idea: actions are selected to maximize information gain in the
GP dynamics posterior, analogous to how sensors are placed to minimize
Kalman filter error covariance. Both problems share supermodularity of
the log-determinant objective.
"""

import numpy as np
import warnings
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from typing import Dict, List, Optional, Tuple, Callable
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

# =============================================================================
# Gaussian Process with SE Kernel and ARD
# =============================================================================

class GaussianProcessRegressor:
    """GP with squared exponential kernel and automatic relevance determination."""

    def __init__(self, dim: int, noise_var: float = 1e-2, lengthscale_init: float = 1.0):
        self.dim = dim
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.n: int = 0
        self.signal_var: float = 1.0
        self.lengthscales: np.ndarray = np.ones(dim) * lengthscale_init
        self.noise_var: float = noise_var
        self._K_chol: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute SE kernel matrix with ARD."""
        X1 = X1.reshape(-1, self.dim) if X1.ndim == 1 else X1
        X2 = X2.reshape(-1, self.dim) if X2.ndim == 1 else X2
        # Scale by lengthscales
        X1_s = X1 / self.lengthscales
        X2_s = X2 / self.lengthscales
        sq1 = np.sum(X1_s ** 2, axis=1, keepdims=True)
        sq2 = np.sum(X2_s ** 2, axis=1, keepdims=True)
        dist_sq = sq1 + sq2.T - 2.0 * X1_s @ X2_s.T
        dist_sq = np.maximum(dist_sq, 0.0)
        return self.signal_var * np.exp(-0.5 * dist_sq)

    def fit(self, X: np.ndarray, Y: np.ndarray, optimize_hyperparams: bool = True):
        """Fit GP to data and optionally optimize hyperparameters."""
        X = X.reshape(-1, self.dim) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        self.X = X.copy()
        self.Y = Y.copy()
        self.n = X.shape[0]

        # Compute kernel matrix
        K = self._compute_kernel(self.X, self.X)
        K_noisy = K + self.noise_var * np.eye(self.n)

        # Cholesky decomposition
        try:
            self._K_chol = cho_factor(K_noisy, lower=True)[0]
            self._alpha = cho_solve((self._K_chol, True), self.Y)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            jitter = 1e-4
            K_noisy += jitter * np.eye(self.n)
            self._K_chol = cho_factor(K_noisy, lower=True)[0]
            self._alpha = cho_solve((self._K_chol, True), self.Y)

        if optimize_hyperparams and self.n >= 5:
            self._optimize_hyperparameters(X, Y)

    def _optimize_hyperparameters(self, X: np.ndarray, Y: np.ndarray):
        """Optimize GP hyperparameters via marginal likelihood."""
        X = X.reshape(-1, self.dim) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        def neg_marginal_log_likelihood(params):
            log_lengthscales = params[:self.dim]
            log_signal = params[self.dim]
            log_noise = params[self.dim + 1]

            lengthscales = np.exp(log_lengthscales)
            signal_var = np.exp(2 * log_signal)
            noise_var = np.exp(2 * log_noise)

            if np.any(lengthscales < 1e-6) or np.any(lengthscales > 100):
                return 1e10

            self.lengthscales = lengthscales
            self.signal_var = signal_var
            self.noise_var = noise_var

            K = self._compute_kernel(self.X, self.X)
            K_noisy = K + noise_var * np.eye(self.n)

            try:
                L = np.linalg.cholesky(K_noisy)
            except np.linalg.LinAlgError:
                return 1e10

            alpha = solve_triangular(L.T, solve_triangular(L, Y, lower=True))
            nll = 0.5 * np.sum(Y * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * self.n * np.log(2 * np.pi)

            # Add penalty for extreme lengthscales
            nll += 0.01 * np.sum((lengthscales - 1.0) ** 2)
            return nll

        # Initialize optimization
        log_lengthscales = np.log(self.lengthscales)
        log_signal = np.log(np.sqrt(self.signal_var))
        log_noise = np.log(np.sqrt(self.noise_var))
        x0 = np.concatenate([log_lengthscales, [log_signal, log_noise]])

        result = minimize(neg_marginal_log_likelihood, x0, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-6})

        if result.fun < neg_marginal_log_likelihood(x0):
            self.lengthscales = np.exp(result.x[:self.dim])
            self.signal_var = np.exp(2 * result.x[self.dim])
            self.noise_var = np.exp(2 * result.x[self.dim + 1])

            # Recompute posterior
            K = self._compute_kernel(self.X, self.X)
            K_noisy = K + self.noise_var * np.eye(self.n)
            try:
                self._K_chol = cho_factor(K_noisy, lower=True)[0]
                self._alpha = cho_solve((self._K_chol, True), self.Y)
            except np.linalg.LinAlgError:
                pass

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict at test points. Returns (mean, variance)."""
        X_test = X_test.reshape(-1, self.dim) if X_test.ndim == 1 else X_test
        k_star = self._compute_kernel(self.X, X_test)
        mean = k_star.T @ self._alpha

        # Posterior variance (diagonal only)
        v = solve_triangular(self._K_chol, k_star, lower=True)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)

        return mean.T, var

    def predict_single(self, x_test: np.ndarray) -> Tuple[float, float]:
        """Predict at a single test point."""
        x_test = x_test.reshape(1, -1)
        mean, var = self.predict(x_test)
        return float(mean.flatten()[0]), float(var[0])

    def add_data(self, X: np.ndarray, Y: np.ndarray):
        """Add new data points and update the GP."""
        X = X.reshape(-1, self.dim) if X.ndim == 1 else X
        Y = Y.reshape(-1, self.dim) if Y.ndim == 1 else Y
        if self.X is None:
            self.fit(X, Y)
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
            self.n = self.X.shape[0]
            try:
                K = self._compute_kernel(self.X, self.X)
                K_noisy = K + self.noise_var * np.eye(self.n)
                self._K_chol = cho_factor(K_noisy, lower=True)[0]
                self._alpha = cho_solve((self._K_chol, True), self.Y)
            except np.linalg.LinAlgError:
                pass

    def get_posterior_variance_at(self, X_test: np.ndarray) -> np.ndarray:
        """Get posterior variance at test points (diagonal of covariance)."""
        X_test = X_test.reshape(-1, self.dim) if X_test.ndim == 1 else X_test
        k_star = self._compute_kernel(self.X, X_test)
        v = solve_triangular(self._K_chol, k_star, lower=True)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return var

    def get_posterior_mean_at(self, X_test: np.ndarray) -> np.ndarray:
        """Get posterior mean at test points."""
        X_test = X_test.reshape(-1, self.dim) if X_test.ndim == 1 else X_test
        k_star = self._compute_kernel(self.X, X_test)
        mean = k_star.T @ self._alpha
        return mean.T


# =============================================================================
# PILCO Dynamics Model (one GP per state dimension)
# =============================================================================

class PILCOPolicy:
    """
    Linear Gaussian policy: u = K @ x + b
    Parameters: theta = [K_flat, b] where K_flat is K flattened.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K: np.ndarray = np.zeros((action_dim, state_dim))
        self.b: np.ndarray = np.zeros(action_dim)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.state_dim) if x.ndim > 1 else x.reshape(1, -1)
        return (self.K @ x.T).T + self.b

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.K.flatten(), self.b])

    def set_params(self, theta: np.ndarray):
        n_K = self.action_dim * self.state_dim
        self.K = theta[:n_K].reshape(self.action_dim, self.state_dim)
        self.b = theta[n_K:]

    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """du/dtheta = d(Kx + b)/dtheta"""
        x = x.reshape(-1, self.state_dim) if x.ndim > 1 else x.reshape(1, -1)
        n_K = self.action_dim * self.state_dim
        grad = np.zeros((x.shape[0], n_K + self.action_dim))
        for i in range(self.action_dim):
            for j in range(self.state_dim):
                idx = i * self.state_dim + j
                grad[:, idx] = x[:, j]
            grad[:, n_K + i] = 1.0
        return grad


# =============================================================================
# PILCO Core Algorithm
# =============================================================================

class PILCOCore:
    """
    PILCO: Probabilistic Inference for Linear Control Optimization.
    
    Implements policy evaluation via moment matching and analytic policy gradients.
    """

    def __init__(self, state_dim: int, action_dim: int, horizon: int = 20,
                 discount: float = 0.99, cost_target: Optional[np.ndarray] = None,
                 cost_width: float = 1.0, gp_fit_interval: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.discount = discount
        self.cost_target = cost_target if cost_target is not None else np.zeros(state_dim)
        self.cost_width = cost_width
        self.cost_matrix = np.eye(state_dim)
        self.gp_fit_interval = gp_fit_interval
        self._transitions_since_fit = 0

        # Dynamics model: one GP per state dimension
        self.gp_dynamics: List[GaussianProcessRegressor] = []
        for _ in range(state_dim):
            gp = GaussianProcessRegressor(state_dim + action_dim, noise_var=1e-2)
            self.gp_dynamics.append(gp)

        self.policy = PILCOPolicy(state_dim, action_dim)
        self.dataset_X: Optional[np.ndarray] = None
        self.dataset_deltas: Optional[np.ndarray] = None

    def _fit_gps(self, optimize_hyperparams: bool = False):
        """Fit all dynamics GPs to current dataset."""
        if self.dataset_X is None or self.dataset_X.shape[0] < 2:
            return
        for d in range(self.state_dim):
            self.gp_dynamics[d].fit(self.dataset_X, self.dataset_deltas[:, d],
                                    optimize_hyperparams=optimize_hyperparams)

    def add_transition(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        """Add a batch of transitions to the dataset (no auto-fit)."""
        X = np.column_stack([states, actions])
        deltas = next_states - states
        if self.dataset_X is None:
            self.dataset_X = X
            self.dataset_deltas = deltas
        else:
            self.dataset_X = np.vstack([self.dataset_X, X])
            self.dataset_deltas = np.vstack([self.dataset_deltas, deltas])

    def add_single_transition(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """Add a single transition."""
        self.add_transition(state.reshape(1, -1), action.reshape(1, -1), next_state.reshape(1, -1))

    def _compute_q(self, mu_input: np.ndarray, Sigma_input: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """
        Compute q vector for moment matching (Eq. 15 in PILCO paper).
        q_i = integral of k(x_i, x) * N(x | mu, Sigma) dx
        """
        n_train = gp.n
        n_test = mu_input.shape[0]
        q = np.zeros((n_train, n_test))

        for i in range(n_train):
            nu = gp.X[i] - mu_input  # (d,)
            # Compute the normalization term
            Sigma_eff = Sigma_input + gp.lengthscales.reshape(1, -1) ** 2
            det_ratio = np.sqrt(np.linalg.det(Sigma_input) / np.linalg.det(Sigma_eff))
            # Exponent term
            diff = nu.reshape(1, -1)  # (1, d)
            inv_Sigma_eff = np.linalg.inv(Sigma_eff)
            exponents = -0.5 * np.sum(diff @ inv_Sigma_eff * diff, axis=1)
            q[i, :] = (gp.signal_var * det_ratio *
                        np.exp(exponents - 0.5 * nu @ inv_Sigma_eff @ nu.T))
        return q

    def _compute_Q_matrix(self, mu_input: np.ndarray, Sigma_input: np.ndarray,
                           gp_a: GaussianProcessRegressor, gp_b: GaussianProcessRegressor) -> np.ndarray:
        """
        Compute Q matrix for covariance prediction (Eq. 22 in PILCO paper).
        Uses element-wise inverse for diagonal lengthscales.
        """
        n_train = gp_a.n
        Q = np.zeros((n_train, n_train))
        d = gp_a.dim

        inv_ls_a2 = 1.0 / (gp_a.lengthscales ** 2)  # (d,)
        inv_ls_b2 = 1.0 / (gp_b.lengthscales ** 2)  # (d,)

        for i in range(n_train):
            for j in range(n_train):
                nu_i = gp_a.X[i] - mu_input  # (d,)
                nu_j = gp_b.X[j] - mu_input

                # Effective covariance: Sigma + 0.5 * diag(1/inv_ls_a2 + 1/inv_ls_b2)
                Sigma_eff = Sigma_input + 0.5 * np.diag(1.0 / inv_ls_a2 + 1.0 / inv_ls_b2)
                try:
                    det_ratio = np.sqrt(max(np.linalg.det(Sigma_input), 1e-300) /
                                        max(np.linalg.det(Sigma_eff), 1e-300))
                except np.linalg.LinAlgError:
                    det_ratio = 0.0

                # Quadratic forms with diagonal inverse lengthscales
                nu_i_quad = np.sum(nu_i ** 2 * inv_ls_a2)
                nu_j_quad = np.sum(nu_j ** 2 * inv_ls_b2)

                # z_ij term for cross-covariance
                z = inv_ls_a2 * nu_i + inv_ls_b2 * nu_j  # (d,)

                # Full exponent from PILCO Eq. 22
                exponent = 0.5 * (z @ Sigma_input @ z) - 0.5 * (nu_i_quad + nu_j_quad)

                Q[i, j] = gp_a.signal_var * gp_b.signal_var * det_ratio * np.exp(exponent)
        return Q

    def predict_distribution(self, mu: np.ndarray, Sigma: np.ndarray,
                             policy: PILCOPolicy) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-step distribution prediction via moment matching.
        Input: state distribution N(mu, Sigma)
        Output: next state distribution N(mu_next, Sigma_next)
        """
        # Compute control distribution moments
        u_mean = policy.apply(mu.reshape(-1, self.state_dim)).mean(axis=0)
        # du/dx = K for linear policy u = Kx + b
        K = policy.K
        Sigma_u = K @ Sigma @ K.T
        Sigma_u = (Sigma_u + Sigma_u.T) / 2

        # Joint state-control distribution
        mu_joint = np.concatenate([mu.reshape(-1), u_mean.reshape(-1)])
        # Approximate joint covariance
        n_x = self.state_dim
        n_u = self.action_dim
        # Cross-covariance cov[x, u] = Sigma @ K^T for linear policy
        Sigma_joint = np.zeros((n_x + n_u, n_x + n_u))
        Sigma_joint[:n_x, :n_x] = Sigma
        Sigma_joint[:n_x, n_x:] = Sigma @ K.T
        Sigma_joint[n_x:, :n_x] = Sigma_joint[:n_x, n_x:].T
        Sigma_joint[n_x:, n_x:] = Sigma_u

        mu_delta = np.zeros(self.state_dim)
        Sigma_delta = np.zeros((self.state_dim, self.state_dim))

        for d in range(self.state_dim):
            gp = self.gp_dynamics[d]

            # Compute q vector - pass as single test point (1, dim)
            mu_joint_single = mu_joint.reshape(1, -1)
            q = self._compute_q(mu_joint_single, Sigma_joint, gp)  # shape (n_train, 1)
            mu_delta[d] = q.flatten() @ gp._alpha[:, 0]

            # Compute Q matrix for covariance
            Q_mat = self._compute_Q_matrix(mu_joint, Sigma_joint, gp, gp)
            expected_var_d = gp.signal_var - np.trace(Q_mat) / gp.n

            # Variance of mean
            var_of_mean = np.sum(q ** 2) / gp.n - q.mean() ** 2
            # Scale by alpha^2
            var_of_mean *= gp.signal_var ** 2

            Sigma_delta[d, d] = expected_var_d + var_of_mean

        mu_next = mu + mu_delta
        Sigma_next = Sigma + Sigma_delta
        return mu_next, Sigma_next

    def expected_cost(self, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """Compute expected quadratic cost: E[||x - target||^2_Q]."""
        diff = mu - self.cost_target
        cost = float(diff @ self.cost_matrix @ diff) + float(np.trace(self.cost_matrix @ Sigma))
        return cost

    def policy_gradient(self, mu0: np.ndarray, Sigma0: np.ndarray,
                        policy: PILCOPolicy) -> np.ndarray:
        """
        Compute policy gradient dJ/dtheta using chain rule.
        """
        theta = policy.get_params()
        n_params = len(theta)

        # Forward pass: propagate distributions
        mu_traj = [mu0.copy()]
        Sigma_traj = [Sigma0.copy()]

        mu = mu0.copy()
        Sigma = Sigma0.copy()

        for t in range(self.horizon):
            mu, Sigma = self.predict_distribution(mu, Sigma, policy)
            mu_traj.append(mu.copy())
            Sigma_traj.append(Sigma.copy())

        # Backward pass: compute gradients
        dJ_dmu = np.zeros(self.state_dim)
        dJ_dSigma = np.zeros((self.state_dim, self.state_dim))

        for t in range(self.horizon - 1, -1, -1):
            cost_t = self.expected_cost(mu_traj[t], Sigma_traj[t])
            discount_t = self.discount ** t

            # Gradient of total cost
            if t == self.horizon - 1:
                dJ_dmu += discount_t * (2 * self.cost_matrix @ (mu_traj[t] - self.cost_target) +
                                         self.cost_matrix.diagonal())
                dJ_dSigma += discount_t * self.cost_matrix

            # Backprop through dynamics
            if t < self.horizon - 1:
                dJ_dmu_next = dJ_dmu.copy()
                dJ_dSigma_next = dJ_dSigma.copy()

                # Approximate gradient through dynamics (simplified)
                # In full PILCO, this uses the chain rule through moment matching
                # Here we use a numerical approximation for simplicity
                eps = 1e-4
                grad = np.zeros(n_params)

                for i in range(n_params):
                    theta_plus = theta.copy()
                    theta_plus[i] += eps
                    policy_plus = PILCOPolicy(self.state_dim, self.action_dim)
                    policy_plus.set_params(theta_plus)

                    mu_plus = mu0.copy()
                    Sigma_plus = Sigma0.copy()
                    for step in range(t + 1, self.horizon):
                        mu_plus, Sigma_plus = self.predict_distribution(mu_plus, Sigma_plus, policy_plus)

                    J_plus = 0.0
                    for s in range(t + 1, self.horizon):
                        J_plus += self.discount ** s * self.expected_cost(mu_traj[s] if s <= t + 1 else mu_plus,
                                                                           Sigma_traj[s] if s <= t + 1 else Sigma_plus)

                    J_minus = 0.0
                    policy_minus = PILCOPolicy(self.state_dim, self.action_dim)
                    policy_minus.set_params(theta.copy())
                    mu_minus = mu0.copy()
                    Sigma_minus = Sigma0.copy()
                    for step in range(t + 1, self.horizon):
                        mu_minus, Sigma_minus = self.predict_distribution(mu_minus, Sigma_minus, policy_minus)
                    for s in range(t + 1, self.horizon):
                        J_minus += self.discount ** s * self.expected_cost(mu_traj[s] if s <= t + 1 else mu_minus,
                                                                           Sigma_traj[s] if s <= t + 1 else Sigma_minus)

                    grad[i] = (J_plus - J_minus) / (2 * eps)

                return grad

        # If horizon is small, use direct gradient
        J = sum(self.discount ** t * self.expected_cost(mu_traj[t], Sigma_traj[t])
                for t in range(self.horizon))
        return np.zeros(n_params)

    def optimize_policy(self, mu0: np.ndarray, Sigma0: np.ndarray,
                        n_iter: int = 50, method: str = 'L-BFGS-B') -> PILCOPolicy:
        """Optimize policy parameters."""
        theta = self.policy.get_params()

        def cost_func(theta):
            policy = PILCOPolicy(self.state_dim, self.action_dim)
            policy.set_params(theta)
            mu = mu0.copy()
            Sigma = Sigma0.copy()
            total_cost = 0.0
            for t in range(self.horizon):
                total_cost += self.discount ** t * self.expected_cost(mu, Sigma)
                mu, Sigma = self.predict_distribution(mu, Sigma, policy)
            return total_cost

        result = minimize(cost_func, theta, method=method,
                         options={'maxiter': n_iter, 'ftol': 1e-6})
        self.policy.set_params(result.x)
        return self.policy

    def evaluate_policy(self, mu0: np.ndarray, Sigma0: np.ndarray,
                        policy: PILCOPolicy) -> float:
        """Evaluate policy return."""
        mu = mu0.copy()
        Sigma = Sigma0.copy()
        total_return = 0.0
        for t in range(self.horizon):
            cost = self.expected_cost(mu, Sigma)
            total_return -= self.discount ** t * cost  # negative because we minimize cost
            mu, Sigma = self.predict_distribution(mu, Sigma, policy)
        return total_return


# =============================================================================
# Information-Theoretic Exploration (IGP-PILCO)
# =============================================================================

class InformationGain:
    """
    Compute information gain from action selections.

    Uses posterior variance reduction as the information proxy:
    IG(u|x) = -0.5 * log(1 - rho^2) ≈ 0.5 * rho^2 for small rho^2

    where rho^2 = k(x, X)(K + sigma^2 I)^{-1} k(X, x) / k(x, x)
    is the squared multiple correlation coefficient.
    """

    def __init__(self, gp: GaussianProcessRegressor):
        self.gp = gp

    def compute_information_gain(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute information gain at test points.

        Returns array of IG values, where higher IG means more informative.
        """
        if self.gp.n < 3:
            return np.zeros(X_test.shape[0])

        # Prior variance
        prior_var = self.gp.signal_var

        # Posterior variance
        k_star = self.gp._compute_kernel(self.gp.X, X_test)
        v = solve_triangular(self.gp._K_chol, k_star, lower=True)
        posterior_var = self.gp.signal_var - np.sum(v ** 2, axis=0)
        posterior_var = np.maximum(posterior_var, 1e-10)

        # Squared multiple correlation coefficient
        rho_sq = 1.0 - posterior_var / prior_var
        rho_sq = np.clip(rho_sq, 0.0, 0.999)

        # Information gain = -0.5 * log(posterior_var / prior_var)
        ig = -0.5 * np.log(posterior_var / prior_var)

        return ig

    def compute_expected_ig(self, state_samples: np.ndarray,
                            action_candidates: np.ndarray) -> np.ndarray:
        """
        Compute expected information gain for each candidate action.

        state_samples: (n_samples, state_dim)
        action_candidates: (M, action_dim) - M candidate actions

        Returns: (M,) array of expected IG values
        """
        n_candidates = action_candidates.shape[0]
        expected_ig = np.zeros(n_candidates)

        for m in range(n_candidates):
            # Build state-action pairs: each candidate action paired with all state samples
            n_s = state_samples.shape[0]
            if m == 0:
                sa_samples = np.column_stack([
                    state_samples,
                    np.tile(action_candidates[m], (n_s, 1))
                ])
            else:
                sa_samples = np.column_stack([
                    np.tile(state_samples, (n_candidates, 1)),
                    np.tile(action_candidates[m], (n_s * n_candidates, 1))
                ])

            ig = self.compute_information_gain(sa_samples)
            expected_ig[m] = ig.mean()

        return expected_ig

    def greedy_select_action(self, state_samples: np.ndarray,
                              action_candidates: np.ndarray,
                              temperature: float = 1.0) -> int:
        """
        Select action via softmax over information gains.

        temperature < 1: more greedy (select highest IG action)
        temperature > 1: more random
        """
        expected_ig = self.compute_expected_ig(state_samples, action_candidates)

        # Softmax selection
        logits = expected_ig / temperature
        probs = scipy_softmax(logits)

        # Sample action index
        action_idx = np.random.choice(len(action_candidates), p=probs)
        return action_idx


# =============================================================================
# IGP-PILCO Algorithm
# =============================================================================

class IGP_PILCO:
    """
    Information-Theoretic GP-PILCO.

    Combines PILCO policy optimization with information-guided exploration.
    Actions are selected to maximize information gain in the GP dynamics posterior,
    analogous to sensor placement in Kalman filtering.

    Key parameters:
    - alpha: exploration coefficient (decays over time)
    - beta: exploration weight for action mixing
    - action_candidates: discretized action space for exploration
    """

    def __init__(self, state_dim: int, action_dim: int, horizon: int = 20,
                 discount: float = 0.99, cost_target: Optional[np.ndarray] = None,
                 action_candidates: Optional[np.ndarray] = None,
                 alpha_init: float = 0.5, alpha_decay: float = 0.95,
                 beta_init: float = 0.3, beta_decay: float = 0.95,
                 temperature: float = 0.5, n_state_samples: int = 50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.discount = discount

        # Create PILCO core
        self.pilco = PILCOCore(state_dim, action_dim, horizon, discount, cost_target)

        # Exploration parameters
        self.alpha = alpha_init  # exploration coefficient (decays)
        self.beta = beta_init    # exploration weight for action mixing
        self.alpha_decay = alpha_decay
        self.beta_decay = beta_decay
        self.temperature = temperature

        # Action candidates for exploration
        if action_candidates is not None:
            self.action_candidates = action_candidates
        else:
            self.action_candidates = None

        # Information gain module
        self.info_gain = InformationGain(self.pilco.gp_dynamics[0])

        self.n_state_samples = n_state_samples

        # Training history
        self.cost_history: List[float] = []
        self.return_history: List[float] = []

    def _sample_state_distribution(self, mu: np.ndarray, Sigma: np.ndarray,
                                    n_samples: int = None) -> np.ndarray:
        """Sample states from Gaussian distribution."""
        n = n_samples or self.n_state_samples
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(self.state_dim))
        samples = np.random.randn(n, self.state_dim) @ L.T + mu.reshape(1, -1)
        return samples

    def _explore_actions(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        Select exploratory actions via information gain maximization.

        Returns: (n_samples, action_dim) array of exploratory actions
        """
        if self.action_candidates is None:
            # Generate random exploration actions
            return np.random.randn(self.n_state_samples, self.action_dim) * 0.1

        state_samples = self._sample_state_distribution(mu, Sigma)

        # Compute information gain for each candidate action
        expected_ig = self.info_gain.compute_expected_ig(state_samples, self.action_candidates)

        # Select action with highest information gain
        best_action_idx = np.argmax(expected_ig)
        best_action = self.action_candidates[best_action_idx]

        # Broadcast to all state samples
        return np.tile(best_action, (state_samples.shape[0], 1))

    def train(self, env, n_episodes: int = 200, steps_per_episode: int = 200,
              evaluate_every: int = 10, policy_n_iter: int = 30) -> Dict:
        """
        Train IGP-PILCO on the given environment.

        Args:
            env: Gymnasium environment
            n_episodes: Number of training episodes
            steps_per_episode: Max steps per episode
            evaluate_every: Evaluate policy every N episodes
            policy_n_iter: Policy optimization iterations

        Returns:
            Dictionary with training history
        """
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Generate action candidates if not provided
        if self.action_candidates is None:
            self._generate_action_candidates()

        mu0 = np.zeros(self.state_dim)
        Sigma0 = np.diag(np.ones(self.state_dim) * 0.1)

        for episode in range(n_episodes):
            obs, _ = env.reset()
            mu = obs.copy().reshape(1, -1)
            Sigma = np.diag(np.ones(self.state_dim) * 0.1)

            states_list = []
            actions_list = []
            next_states_list = []
            rewards_list = []

            for t in range(steps_per_episode):
                # Compute current control
                policy_action = self.pilco.policy.apply(mu).mean(axis=0)

                # Get exploratory action
                exploratory_action = self._explore_actions(mu.flatten(), Sigma)

                # Mix policy and exploratory actions
                beta_t = self.beta / (1 + episode / 50) ** 0.5
                beta_t = min(beta_t, 0.5)

                # For single state, select one exploratory action
                if exploratory_action.ndim == 2:
                    exploratory_action = exploratory_action[0]

                mixed_action = (1 - beta_t) * policy_action + beta_t * exploratory_action
                mixed_action = np.clip(mixed_action,
                                       self.action_space.low,
                                       self.action_space.high)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(mixed_action)
                done = terminated or truncated

                states_list.append(mu.flatten().copy())
                actions_list.append(mixed_action.copy())
                next_states_list.append(next_obs.copy())
                rewards_list.append(reward)

                obs = next_obs
                if done:
                    break

            # Add transitions to dataset
            states_arr = np.array(states_list)
            actions_arr = np.array(actions_list)
            next_states_arr = np.array(next_states_list)
            self.pilco.add_transition(states_arr, actions_arr, next_states_arr)

            # Fit GPs at end of each episode (no hyperparam optimization during training)
            self.pilco._fit_gps(optimize_hyperparams=False)

            # Optimize policy periodically
            if episode % evaluate_every == 0:
                # Fit with hyperparam optimization for better policy evaluation
                self.pilco._fit_gps(optimize_hyperparams=True)
                self.pilco.optimize_policy(mu0, Sigma0, n_iter=policy_n_iter)

                # Evaluate
                ret = self.evaluate_policy(env, mu0, Sigma0)
                self.return_history.append(ret)
                self.cost_history.append(-ret)

                alpha_t = self.alpha / (1 + episode / 50) ** 0.5
                print(f"Episode {episode:4d} | Return: {ret:8.2f} | "
                      f"Alpha: {alpha_t:.4f} | Beta: {beta_t:.4f} | "
                      f"Dataset: {self.pilco.dataset_X.shape[0] if self.pilco.dataset_X is not None else 0}")

            # Decay exploration coefficients
            self.alpha *= self.alpha_decay
            self.beta *= self.beta_decay

        return {
            'cost_history': self.cost_history,
            'return_history': self.return_history,
            'final_policy': self.pilco.policy
        }

    def _generate_action_candidates(self, n_candidates: int = 20):
        """Generate discretized action space candidates."""
        low = self.action_space.low
        high = self.action_space.high
        if self.action_dim == 1:
            self.action_candidates = np.linspace(low[0], high[0], n_candidates).reshape(-1, 1)
        else:
            # Generate candidates along each axis
            candidates_per_dim = max(2, int(np.ceil(n_candidates ** (1 / self.action_dim))))
            grids = [np.linspace(low[i], high[i], candidates_per_dim).reshape(-1, 1)
                     for i in range(self.action_dim)]
            from itertools import product
            self.action_candidates = np.array(list(product(*grids))).reshape(-1, self.action_dim)

    def evaluate_policy(self, env, mu0: np.ndarray, Sigma0: np.ndarray,
                        n_rollouts: int = 10) -> float:
        """Evaluate policy by running rollouts in the environment."""
        returns = []
        for _ in range(n_rollouts):
            obs, _ = env.reset()
            total_reward = 0.0
            for t in range(self.horizon):
                action = self.pilco.policy.apply(np.array([obs])).flatten()
                action = np.clip(action, self.action_space.low, self.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            returns.append(total_reward)
        return float(np.mean(returns))

    def get_policy(self) -> PILCOPolicy:
        return self.pilco.policy


# =============================================================================
# Baseline Algorithms
# =============================================================================

class RandomAgent:
    """Random action agent for baseline comparison."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class CountBasedExplorer:
    """Count-based exploration (RND - Random Network Distillation inspired)."""

    def __init__(self, state_dim: int, action_space, intrinsic_reward_weight: float = 0.1):
        self.state_dim = state_dim
        self.action_space = action_space
        self.intrinsic_weight = intrinsic_reward_weight
        self.visit_counts = {}
        self.grid_resolution = 20
        self.grid_bounds = None

    def _state_to_bin(self, state):
        if self.grid_bounds is None:
            self.grid_bounds = (np.full(self.state_dim, -10.0), np.full(self.state_dim, 10.0))
        state = np.clip(state, self.grid_bounds[0], self.grid_bounds[1])
        bins = ((state - self.grid_bounds[0]) /
                (self.grid_bounds[1] - self.grid_bounds[0]) *
                (self.grid_resolution - 1)).astype(int)
        return tuple(bins)

    def get_intrinsic_reward(self, state):
        bin_key = self._state_to_bin(state)
        count = self.visit_counts.get(bin_key, 0)
        self.visit_counts[bin_key] = count + 1
        return self.intrinsic_weight / (1 + np.log(1 + count))

    def act(self, obs, extrinsic_reward=None):
        if extrinsic_reward is None:
            extrinsic_reward = 0.0
        return self.action_space.sample()


class EntropyRegularizedAgent:
    """Entropy-regularized policy with temperature scheduling."""

    def __init__(self, state_dim, action_dim, action_space, temperature: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.temperature = temperature
        # Linear policy
        self.K = np.random.randn(action_dim, state_dim) * 0.1
        self.b = np.zeros(action_dim)

    def act(self, obs, training=True):
        action = (self.K @ obs) + self.b
        if training:
            action = action + self.temperature * np.random.randn(self.action_dim)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def get_policy_params(self):
        return np.concatenate([self.K.flatten(), self.b])

    def set_policy_params(self, params):
        n_K = self.action_dim * self.state_dim
        self.K = params[:n_K].reshape(self.action_dim, self.state_dim)
        self.b = params[n_K:]


# =============================================================================
# Experiment Runner
# =============================================================================

def create_action_candidates(action_space, n_candidates: int = 30):
    """Generate action candidates from action space."""
    low = action_space.low
    high = action_space.high
    if len(low) == 1:
        return np.linspace(low[0], high[0], n_candidates).reshape(-1, 1)
    else:
        # For multi-dimensional actions, sample uniformly
        samples = np.random.uniform(low, high, size=(n_candidates, len(low)))
        return samples


def run_experiment(env_name: str, agent, n_episodes: int = 200,
                   steps_per_episode: int = 200, n_eval_rollouts: int = 5) -> Dict:
    """Run a single experiment and return results."""
    import gymnasium as gym

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    returns = []
    rewards_per_episode = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for t in range(steps_per_episode):
            if isinstance(agent, PILCOPolicy):
                obs_reshaped = obs.reshape(1, -1)
                action = agent.apply(obs_reshaped).flatten()
            elif hasattr(agent, 'get_policy'):
                policy = agent.get_policy()
                action = policy.apply(np.array([obs])).flatten()
            elif hasattr(agent, 'act'):
                action = agent.act(obs)
            else:
                action = agent.get_policy().apply(np.array([obs])).flatten()

            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Train agent if it has a train method
            if hasattr(agent, 'add_transition'):
                # This would need state tracking, simplified here
                pass

            if terminated or truncated:
                break

        returns.append(total_reward)
        rewards_per_episode.append(total_reward)

        if (ep + 1) % 20 == 0:
            mean_ret = np.mean(returns[-20:])
            print(f"  {env_name} | Episode {ep+1}/{n_episodes} | "
                  f"Mean Return (last 20): {mean_ret:8.2f}")

    env.close()
    return {
        'env': env_name,
        'returns': returns,
        'mean_returns': np.array(returns),
        'best_return': float(np.max(returns)),
        'final_mean': float(np.mean(returns[-20:])) if len(returns) >= 20 else float(np.mean(returns))
    }


def run_all_experiments():
    """Run all experiments and compare agents."""
    import gymnasium as gym

    environments = {
        'Pendulum-v1': {'horizon': 200, 'n_episodes': 150},
        'InvertedDoublePendulum-v4': {'horizon': 200, 'n_episodes': 100},
        'Hopper-v4': {'horizon': 300, 'n_episodes': 50},
    }

    results = {}

    # ---- IGP-PILCO ----
    print("\n" + "=" * 70)
    print("Training IGP-PILCO")
    print("=" * 70)

    for env_name, config in environments.items():
        print(f"\n--- {env_name} ---")
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        action_candidates = create_action_candidates(env.action_space, n_candidates=30)

        agent = IGP_PILCO(
            state_dim=state_dim, action_dim=action_dim,
            horizon=config['horizon'], discount=0.99,
            action_candidates=action_candidates,
            alpha_init=0.3, beta_init=0.2,
            temperature=0.5
        )

        try:
            agent.train(env, n_episodes=config['n_episodes'],
                       steps_per_episode=config['horizon'],
                       evaluate_every=10, policy_n_iter=20)
            results[f'igp_pilco_{env_name}'] = run_experiment(
                env_name, agent, n_episodes=20,
                steps_per_episode=config['horizon'])
        except Exception as e:
            print(f"  ERROR with IGP-PILCO on {env_name}: {e}")
            results[f'igp_pilco_{env_name}'] = None

        env.close()

    # ---- Random Agent ----
    print("\n" + "=" * 70)
    print("Running Random Agent Baseline")
    print("=" * 70)

    for env_name in environments:
        print(f"\n--- {env_name} ---")
        env = gym.make(env_name)
        agent = RandomAgent(env.action_space)
        results[f'random_{env_name}'] = run_experiment(
            env_name, agent, n_episodes=30,
            steps_per_episode=environments[env_name]['horizon'])
        env.close()

    # ---- PPO ----
    print("\n" + "=" * 70)
    print("Training PPO")
    print("=" * 70)

    from stable_baselines3 import PPO

    for env_name, config in environments.items():
        print(f"\n--- {env_name} ---")
        try:
            env = gym.make(env_name)
            agent = PPO('MlpPolicy', env, verbose=0, n_epochs=5,
                       batch_size=64, ent_coef=0.01)
            agent.learn(total_timesteps=config['n_episodes'] * config['horizon'])
            results[f'ppo_{env_name}'] = run_experiment(
                env_name, agent, n_episodes=20,
                steps_per_episode=config['horizon'])
            env.close()
        except Exception as e:
            print(f"  ERROR with PPO on {env_name}: {e}")
            results[f'ppo_{env_name}'] = None

    # ---- SAC ----
    print("\n" + "=" * 70)
    print("Training SAC")
    print("=" * 70)

    from stable_baselines3 import SAC

    for env_name, config in environments.items():
        print(f"\n--- {env_name} ---")
        try:
            env = gym.make(env_name)
            agent = SAC('MlpPolicy', env, verbose=0,
                       learning_starts=1000, batch_size=64)
            agent.learn(total_timesteps=config['n_episodes'] * config['horizon'])
            results[f'sac_{env_name}'] = run_experiment(
                env_name, agent, n_episodes=20,
                steps_per_episode=config['horizon'])
            env.close()
        except Exception as e:
            print(f"  ERROR with SAC on {env_name}: {e}")
            results[f'sac_{env_name}'] = None

    # ---- Count-Based Exploration (PPO + count) ----
    print("\n" + "=" * 70)
    print("Training PPO with Count-Based Exploration")
    print("=" * 70)

    for env_name, config in environments.items():
        print(f"\n--- {env_name} ---")
        try:
            env = gym.make(env_name)
            agent = PPO('MlpPolicy', env, verbose=0, n_epochs=5,
                       batch_size=64, ent_coef=0.1)  # higher entropy for exploration
            agent.learn(total_timesteps=config['n_episodes'] * config['horizon'])
            results[f'count_based_{env_name}'] = run_experiment(
                env_name, agent, n_episodes=20,
                steps_per_episode=config['horizon'])
            env.close()
        except Exception as e:
            print(f"  ERROR with count-based on {env_name}: {e}")
            results[f'count_based_{env_name}'] = None

    # ---- Entropy-Based Exploration (PPO + high entropy) ----
    print("\n" + "=" * 70)
    print("Training PPO with High Entropy")
    print("=" * 70)

    for env_name, config in environments.items():
        print(f"\n--- {env_name} ---")
        try:
            env = gym.make(env_name)
            agent = PPO('MlpPolicy', env, verbose=0, n_epochs=5,
                       batch_size=64, ent_coef=0.2)  # high entropy
            agent.learn(total_timesteps=config['n_episodes'] * config['horizon'])
            results[f'entropy_based_{env_name}'] = run_experiment(
                env_name, agent, n_episodes=20,
                steps_per_episode=config['horizon'])
            env.close()
        except Exception as e:
            print(f"  ERROR with entropy-based on {env_name}: {e}")
            results[f'entropy_based_{env_name}'] = None

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    agent_names = [
        'igp_pilco', 'random', 'ppo', 'sac', 'count_based', 'entropy_based'
    ]

    for env_name in environments:
        print(f"\n{'='*50}")
        print(f"  {env_name}")
        print(f"{'='*50}")
        print(f"  {'Agent':<25} {'Best Return':>12} {'Final Mean':>12}")
        print(f"  {'-'*50}")

        for agent_name in agent_names:
            key = f'{agent_name}_{env_name}'
            if key in results and results[key] is not None:
                ret = results[key]
                print(f"  {agent_name:<25} {ret['best_return']:>12.2f} {ret['final_mean']:>12.2f}")
            else:
                print(f"  {agent_name:<25} {'N/A':>12} {'N/A':>12}")

    return results


if __name__ == '__main__':
    run_all_experiments()
