import torch
from typing import Callable, List, Optional, Tuple
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood


class VariationalGP(ApproximateGP):
    """
    Variational GP for sparse approximation when we have many points.
    """
    def __init__(self, train_X: torch.Tensor, num_inducing: int = 50):
        # Initialize inducing points from a subset of training data
        n_train = train_X.shape[0]
        num_inducing = min(num_inducing, n_train)

        # Select inducing points (use first num_inducing points, or sample uniformly)
        if n_train <= num_inducing:
            inducing_points = train_X
        else:
            # Uniformly sample inducing points
            indices = torch.linspace(0, n_train - 1, num_inducing, dtype=torch.long, device=train_X.device)
            inducing_points = train_X[indices]

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(VariationalGP, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return torch.distributions.MultivariateNormal(mean_x, covar_x)


class Surrogate:
    """
    Surrogate model that switches between GP and sparse GP based on number of points.

    Maintains a record of all points and filters by bounds when fitting.

    Parameters
    ----------
    max_points : int
        Maximum number of points before switching to sparse GP. Default: 100.
    num_inducing : int
        Number of inducing points for sparse GP. Default: 50.
    device : str
        Device for computations. Default: 'cuda'.
    dtype : torch.dtype
        Data type. Default: torch.float64.
    """
    def __init__(self, max_points: int = 100, num_inducing: int = 50,
                 device: str = 'cuda', dtype: torch.dtype = torch.float64):
        self.max_points = max_points
        self.num_inducing = num_inducing
        self.device = torch.device(device)
        self.dtype = dtype

        # Maintain record of ALL points (not just those in current bounds)
        self.X_all = torch.tensor([], device=self.device, dtype=self.dtype)
        self.Y_all = torch.tensor([], device=self.device, dtype=self.dtype)

        self.model = None
        self.mll = None
        self.bounds = None

    def _points_in_bounds(self, X: torch.Tensor, bounds: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Check which points are within the given bounds.

        Parameters
        ----------
        X : torch.Tensor
            Points to check (N, d).
        bounds : torch.Tensor, optional
            Bounds (2, d) with [lower, upper]. If None, all points are considered in bounds.

        Returns
        -------
        torch.Tensor
            Boolean mask (N,) indicating which points are in bounds.
        """
        if bounds is None:
            return torch.ones(X.shape[0], dtype=torch.bool, device=X.device)

        # Ensure bounds has correct shape (2, d)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")

        lower, upper = bounds[0], bounds[1]
        # Ensure broadcasting works correctly: X is (N, d), lower/upper are (d,)
        in_bounds = torch.all((X >= lower.unsqueeze(0)) & (X <= upper.unsqueeze(0)), dim=1)
        return in_bounds

    def update(self, X: torch.Tensor, Y: torch.Tensor, bounds: Optional[torch.Tensor] = None):
        """
        Update the surrogate with new data.

        Maintains all points in X_all and Y_all, but only fits the model
        using points within the current bounds.

        Parameters
        ----------
        X : torch.Tensor
            New input points (N, d).
        Y : torch.Tensor
            New output values (N,).
        bounds : torch.Tensor, optional
            Current bounds (2, d) for filtering points when fitting.
        """
        X = X.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        # Ensure Y is 1D
        if Y.ndim > 1:
            Y = Y.squeeze()

        # Add to the record of all points
        self.X_all = torch.cat([self.X_all, X], dim=0) if self.X_all.numel() > 0 else X
        self.Y_all = torch.cat([self.Y_all, Y], dim=0) if self.Y_all.numel() > 0 else Y

        # Store bounds for later use
        if bounds is not None:
            bounds = bounds.to(device=self.device, dtype=self.dtype)
            # Validate bounds shape (2, d)
            if bounds.ndim != 2 or bounds.shape[0] != 2:
                raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")
            self.bounds = bounds

        # Fit the model using points within bounds
        self._fit_model()

    def _fit_model(self):
        """Fit the GP model using points within the current bounds."""
        if self.X_all.numel() == 0:
            return

        # Filter points within bounds
        if self.bounds is not None:
            in_bounds_mask = self._points_in_bounds(self.X_all, self.bounds)
            X_fit = self.X_all[in_bounds_mask]
            Y_fit = self.Y_all[in_bounds_mask]
        else:
            X_fit = self.X_all
            Y_fit = self.Y_all

        # Need at least 2 points to fit a GP
        if X_fit.shape[0] < 2:
            return

        # Reshape Y for GP (needs to be (N, 1))
        if Y_fit.ndim == 1:
            Y_fit = Y_fit.unsqueeze(1)

        # Determine number of points in bounds
        n_points = X_fit.shape[0]

        # Switch between GP and sparse GP based on max_points
        if n_points < self.max_points:
            # Use regular GP
            self.model = SingleTaskGP(X_fit, Y_fit)
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(self.mll)
        else:
            # Use sparse GP (variational)
            # Use min of num_inducing and actual number of points
            num_inducing = min(self.num_inducing, n_points)

            # Initialize variational GP
            self.model = VariationalGP(X_fit, num_inducing=num_inducing)
            self.model = self.model.to(device=self.device, dtype=self.dtype)

            # Use VariationalELBO for fitting
            self.mll = VariationalELBO(
                self.model.likelihood,
                self.model,
                num_data=X_fit.shape[0]
            )
            self.mll = self.mll.to(device=self.device, dtype=self.dtype)

            # Fit the model
            self.model.train()
            self.model.likelihood.train()

            optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.model.likelihood.parameters()},
            ], lr=0.01)

            # Training loop for variational GP
            for _ in range(50):  # 50 iterations should be sufficient
                optimizer.zero_grad()
                output = self.model(X_fit)
                loss = -self.mll(output, Y_fit.squeeze())
                loss.backward()
                optimizer.step()

            self.model.eval()
            self.model.likelihood.eval()

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the surrogate at given points.

        Parameters
        ----------
        X : torch.Tensor
            Query points (N, d) or (d,).

        Returns
        -------
        torch.Tensor
            Predicted values (N,) or scalar.
        """
        if self.model is None:
            # If no model, return zeros or mean of observed values
            if self.Y_all.numel() > 0:
                return torch.full((X.shape[0] if X.ndim > 1 else 1,),
                                 self.Y_all.mean().item(),
                                 device=self.device, dtype=self.dtype)
            else:
                return torch.zeros(X.shape[0] if X.ndim > 1 else 1,
                                  device=self.device, dtype=self.dtype)

        X = X.to(device=self.device, dtype=self.dtype)

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.unsqueeze(0)
            was_1d = True
        else:
            was_1d = False

        self.model.eval()
        if isinstance(self.model, VariationalGP):
            # For GPyTorch VariationalGP, use likelihood for predictions
            self.model.likelihood.eval()
            with torch.no_grad():
                f_dist = self.model(X)
                f_mean = f_dist.mean
                # Apply likelihood to get predictive distribution
                predictive = self.model.likelihood(f_dist)
                mean = predictive.mean.squeeze()
        else:
            # For BoTorch SingleTaskGP, use posterior
            with torch.no_grad():
                posterior = self.model.posterior(X)
                mean = posterior.mean.squeeze()

        if was_1d:
            return mean.squeeze(0) if mean.ndim > 0 else mean
        return mean


class ArcherfishObjective:
    """
    Objective wrapper that uses surrogates to fill in missing values.

    Parameters
    ----------
    objective : Callable
        Objective function that takes (endpoints, bounds) and returns (x_actual, y).
    surrogate_idxs : List[int]
        Indices where surrogates should be used (all values < n where n is number of metrics).
    weights : List[float]
        Weights for each metric (length n).
    max_points : int
        Maximum points before switching to sparse GP. Default: 100.
    num_inducing : int
        Number of inducing points for sparse GP. Default: 50.
    device : str
        Device for computations. Default: 'cuda'.
    dtype : torch.dtype
        Data type. Default: torch.float64.
    """
    def __init__(self, objective: Callable, surrogate_idxs: List[int], weights: List[float],
                 max_points: int = 100, num_inducing: int = 50,
                 device: str = 'cuda', dtype: torch.dtype = torch.float64):
        self.objective = objective
        self.surrogate_idxs = surrogate_idxs  # all values < n where n is the number of different metrics
        self.weights = weights  # should be a list of length n
        self.device = torch.device(device)
        self.dtype = dtype

        # Create surrogates for all specified indices
        self.surrogates = {
            idx: Surrogate(max_points=max_points, num_inducing=num_inducing,
                          device=device, dtype=dtype)
            for idx in surrogate_idxs
        }

    def evaluate(self, endpoints: torch.Tensor, bounds: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate objective and fill in missing values using surrogates.

        Parameters
        ----------
        endpoints : torch.Tensor
            Endpoints (N, d).
        bounds : torch.Tensor, optional
            Bounds (2, d) with [lower, upper]. If None, no bounds filtering is applied.

        Returns
        -------
        tuple
            (x_actual, y_output) where y_output has missing values filled by surrogates.
        """
        endpoints = endpoints.to(device=self.device, dtype=self.dtype)
        if bounds is not None:
            bounds = bounds.to(device=self.device, dtype=self.dtype)
            # Validate bounds shape
            if bounds.ndim != 2 or bounds.shape[0] != 2:
                raise ValueError(f"bounds must have shape (2, d), got {bounds.shape}")

        # Call the objective function
        result = self.objective(endpoints, bounds)
        if isinstance(result, tuple):
            x_actual, y = result
        else:
            raise ValueError("Objective must return (x_actual, y) tuple")

        # Convert to tensors if needed
        if not isinstance(x_actual, torch.Tensor):
            x_actual = torch.tensor(x_actual, device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)

        # Ensure correct device and dtype
        x_actual = x_actual.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        # Ensure y is 2D (N, n_metrics)
        if y.ndim == 1:
            y = y.unsqueeze(1) if len(self.weights) == 1 else y.unsqueeze(0)

        n_metrics = y.shape[1]
        y_output = torch.zeros(x_actual.shape[0], len(self.weights),
                              device=self.device, dtype=self.dtype)

        # Process each surrogate index
        for idx in self.surrogate_idxs:
            if idx >= n_metrics:
                continue

            surrogate = self.surrogates[idx]
            y_col = y[:, idx]

            # Identify valid (non-NaN, non-inf) values
            valid_mask = torch.isfinite(y_col)
            invalid_mask = ~valid_mask

            # Update surrogate with valid data
            if valid_mask.any():
                x_valid = x_actual[valid_mask]
                y_valid = y_col[valid_mask]
                surrogate.update(x_valid, y_valid, bounds)

            # Fill in invalid values with surrogate predictions
            if invalid_mask.any():
                x_invalid = x_actual[invalid_mask]
                y_pred = surrogate.evaluate(x_invalid)
                y_col = y_col.clone()
                y_col[invalid_mask] = y_pred
                y[:, idx] = y_col

            # Apply weight to the corresponding values
            y_output[:, idx] = y_col * self.weights[idx]

        # Fill in non-surrogate indices with original values (with weights)
        for idx in range(len(self.weights)):
            if idx not in self.surrogate_idxs and idx < n_metrics:
                y_output[:, idx] = y[:, idx] * self.weights[idx]

        return x_actual, y_output
