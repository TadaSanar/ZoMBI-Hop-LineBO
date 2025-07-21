import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import gc

class GPUApproxGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, num_inducing=512):
        """
        Sparse GP model using inducing points for efficiency.

        Args:
            train_x: Training inputs (torch tensor)
            train_y: Training targets (torch tensor)
            likelihood: GPyTorch likelihood
            num_inducing: Number of inducing points (default: 512)
        """
        # Get dtype and device from input tensors
        dtype = train_x.dtype
        device = train_x.device

        # Select inducing points
        if train_x.size(0) > num_inducing:
            # Simple random selection - can be improved with k-means++
            perm = torch.randperm(train_x.size(0))[:num_inducing]
            inducing_points = train_x[perm].clone()
        else:
            inducing_points = train_x.clone()

        # Initialize variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), dtype=dtype
        )

        # Initialize variational strategy without explicit jitter
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        # Call parent's init first to set up parameters
        super().__init__(variational_strategy)

        # Initialize mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean(dtype=dtype)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(dtype=dtype)
        )

        # Store training data
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood

        # Ensure all variational parameters have correct dtype
        for param in self.variational_strategy.parameters():
            param.data = param.data.to(dtype=dtype)

        # Ensure inducing points have correct dtype
        self.variational_strategy.inducing_points = self.variational_strategy.inducing_points.to(dtype=dtype)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, X, y):
        """
        Fit the GP model to new data.

        Args:
            X: Training inputs (numpy array or pandas DataFrame)
            y: Training targets (numpy array or pandas DataFrame)
        """
        # Convert inputs to torch tensors
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Ensure y is 1D
        if y.ndim == 2:
            y = y.squeeze(-1)

        # Get device and dtype from model
        device = next(self.parameters()).device
        torch_dtype = next(self.parameters()).dtype

        # Convert to tensors and move to device
        self.train_x = torch.tensor(X, dtype=torch_dtype, device=device)
        self.train_y = torch.tensor(y, dtype=torch_dtype, device=device)

        # Set to training mode
        self.train()
        self.likelihood.train()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        # "Loss" for GPs - the variational ELBO
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.train_y.size(0))

        # Training loop
        for i in range(50):  # 50 iterations should be sufficient for most cases
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

        return self
