# NOTE TO ALL FUTURE EDITORS:
# This code requires PyTorch and GPyTorch to be installed for the GPU-accelerated GP.
# There is NO REASON to ever put in a fallback mechanism.
# Whoever runs this should have torch installed: pip install torch gpytorch

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from scipy.stats import qmc
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import warnings
from scipy.spatial.distance import cdist
from gp_cuda import GPUExactGP
import torch
import gpytorch
from scipy.spatial.distance import pdist

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def project_to_simplex(v):
    """
    Project a point onto the unit simplex (sum=1, all components >= 0).

    This is a standalone function for use in initialization and other utilities.

    Parameters:
    -----------
    v : array-like, shape (d,)
        Input vector to project

    Returns:
    --------
    array, shape (d,)
        Projected vector on the simplex
    """
    v = np.array(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v

    # Sort in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, len(u) + 1)
    cond = u - cssv / ind > 0

    if np.any(cond):
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
    else:
        theta = 0

    w = np.maximum(v - theta, 0)
    return w


class ZoMBIHop:
    def __init__(self, objective_function, dimensions, X_init_actual, Y_init, num_activations=3,
                 max_gp_points=200, penalty_zoom_percentage=0.8,
                 max_iterations=20, tolerance=0.02,
                 top_m_points=4, resolution=15, max_zoom_levels=4, bounds=None,
                 penalty_batch_size=10000, max_chunk_size=10000000, max_gpu_percentage=0.3,
                 num_experiments=20, linebo_num_lines=50, linebo_pts_per_line=100):
        """
        ZoMBI-Hop: Multi-activation optimization with adaptive penalization.

        This implementation includes optimized penalty mask computation using torch tensors
        and chunking for efficient handling of large point sets (e.g., mesh grids).

        Args:
            objective_function: Function to optimize (minimize). Takes 2D array (n_experiments, dimensions)
                              Returns 2D array (n_experiments, dimensions) and 1D array (n_experiments,)
            dimensions: Number of input dimensions
            X_init_actual: Initial dataset actual experimental points (shape: n_points x dimensions)
            Y_init: Initial dataset output values (shape: n_points,)
            num_activations: Number of activations (needles to find)
            max_gp_points: Maximum points to keep in GP memory
            penalty_zoom_percentage: Percentage of zoom levels to use for penalty area (0.0-1.0)
            max_iterations: Maximum iterations per zoom level
            tolerance: Convergence tolerance for needle detection
            top_m_points: Number of best points for bound computation
            resolution: Mesh resolution for each dimension
            max_zoom_levels: Maximum zoom levels per activation
            bounds: List of (min, max) tuples for each dimension
            penalty_batch_size: Batch size for penalty mask computation (GPU memory management).
                              Larger values use more memory but may be faster. Default: 10000.
            max_chunk_size: Maximum chunk size for GPU processing operations. Default: 10M points.
            max_gpu_percentage: Maximum percentage of GPU memory to use for chunked operations (0.0-1.0).
                               Default: 0.3 (30% of available GPU memory).
            num_experiments: Number of experimental points to sample along best line per acquisition
            linebo_num_lines: Number of candidate lines to evaluate in LineBO
            linebo_pts_per_line: Number of points to sample per line for acquisition integration

        Performance Features:
            - Optimized penalty mask computation using torch tensors with GPU acceleration
            - Chunked processing for large point sets to manage memory usage
            - Efficient vectorized distance calculations using broadcasting
            - Adaptive memory management based on available GPU memory
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.num_activations = num_activations
        self.max_gp_points = max_gp_points
        self.penalty_zoom_percentage = penalty_zoom_percentage
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.top_m_points = top_m_points
        self.resolution = resolution
        self.max_zoom_levels = max_zoom_levels
        self.penalty_batch_size = penalty_batch_size
        self.max_chunk_size = max_chunk_size
        self.max_gpu_percentage = max_gpu_percentage
        self.num_experiments = num_experiments
        self.linebo_num_lines = linebo_num_lines
        self.linebo_pts_per_line = linebo_pts_per_line

        # Set bounds
        if bounds is None:
            self.bounds = [(0, 1) for _ in range(dimensions)]
        else:
            self.bounds = bounds

        # Initialize storage for hop results
        self.needles = []  # Best points found in each activation
        self.needle_values = []  # Best values found in each activation
        self.penalty_regions = []  # Penalty regions around each needle
        self.activation_histories = []  # Detailed history for each activation

        # Streamlined data collection - only track real evaluations
        self.X_all_actual = np.empty((0, dimensions))      # All actual sampled points (real evaluations only)
        self.Y_all = np.empty(0)                          # All objective function values (real evaluations only)
        self.all_penalized = np.array([], dtype=bool)     # Mask: True if point was in penalized region when sampled

        # Evaluation counters
        self.real_eval_count = 0      # Count of real objective function evaluations

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using GPU-accelerated GP on {self.device}")

        self.gp = GPUExactGP(
            train_x=torch.empty(0, dimensions, device=self.device),
            train_y=torch.empty(0, device=self.device),
            likelihood=gpytorch.likelihoods.GaussianLikelihood()
        )
        self.gp = self.gp.to(self.device)
        self.gp.likelihood = self.gp.likelihood.to(self.device)

        # Activation-specific data storage
        self.activation_X_data = []  # Actual experimental points for current activation
        self.activation_Y_data = []  # Experimental results for current activation
        self.convergence_errors = []
        self.bounds_history = []  # Track bounds at each zoom level
        self.zoom_level_markers = []  # Track which points belong to which zoom level

        # Initialize global dataset with provided initialization data
        X_init_actual = np.array(X_init_actual)
        Y_init = np.array(Y_init)

        # Validate input shapes
        if X_init_actual.ndim != 2:
            raise ValueError(f"X_init_actual must be 2D array, got shape {X_init_actual.shape}")
        if Y_init.ndim != 1:
            raise ValueError(f"Y_init must be 1D array, got shape {Y_init.shape}")
        if X_init_actual.shape[0] != Y_init.shape[0]:
            raise ValueError(f"X_init_actual and Y_init must have same number of points: {X_init_actual.shape[0]} vs {Y_init.shape[0]}")
        if X_init_actual.shape[1] != self.dimensions:
            raise ValueError(f"X_init_actual must have {self.dimensions} dimensions, got {X_init_actual.shape[1]}")

        print(f"🎲 Initializing with {len(X_init_actual)} provided data points...")

        # Store initialization data
        self.X_all_actual = X_init_actual.copy()
        self.Y_all = Y_init.copy()

        # No penalization for initial samples (no needles found yet)
        self.all_penalized = np.zeros(len(X_init_actual), dtype=bool)

        # Count initialization as real evaluations
        self.real_eval_count = len(X_init_actual)

        print(f"✅ Initial dataset created:")
        print(f"   Initial points: {len(X_init_actual)}")
        print(f"   Actual evaluations: {len(self.Y_all)}")
        print(f"   Real evaluation count: {self.real_eval_count}")
        print(f"   Best initial value: {np.min(self.Y_all):.6f}")
        print(f"   Average noise estimate: {self._get_average_noise():.6f}")

    def objective_function_wrapper(self, line_endpoints):
        """
        LineBO-compatible wrapper that receives line endpoints and returns non-penalized experimental points.

        Args:
            line_endpoints: 2D numpy array of shape (2, dimensions) containing line start and end points

        Returns:
            x_actual_filtered: 2D numpy array of non-penalized experimental points
            y_actual_filtered: 1D numpy array of corresponding objective values
        """
        # Call objective function with line endpoints
        x_actual_array, y_actual_array = self.objective_function(line_endpoints)

        # Check which points are in penalized regions
        penalty_status = []
        if len(self.needles) > 0:
            penalty_mask = self._compute_penalized_mask(x_actual_array)
            penalty_status = penalty_mask < 0.5  # True if penalized
        else:
            penalty_status = [False] * len(x_actual_array)  # No penalty if no needles yet

        # Add ALL experimental points to global tracking (penalized and non-penalized)
        for i, (x_actual, y_actual, is_penalized) in enumerate(zip(x_actual_array, y_actual_array, penalty_status)):
            # Update global tracking arrays - no more X_requested since we work with line endpoints
            self.X_all_actual = np.vstack([self.X_all_actual, x_actual.reshape(1, -1)]) if len(self.X_all_actual) > 0 else x_actual.reshape(1, -1)

            self.Y_all = np.append(self.Y_all, y_actual)
            self.all_penalized = np.append(self.all_penalized, is_penalized)
            self.real_eval_count += 1

        # Filter out penalized points for return to GP training
        non_penalized_mask = ~np.array(penalty_status)
        x_actual_filtered = x_actual_array[non_penalized_mask]
        y_actual_filtered = y_actual_array[non_penalized_mask]

        if len(x_actual_filtered) == 0:
            print("⚠️  Warning: All experimental points were penalized! Returning empty batch.")
        elif len(x_actual_filtered) < len(x_actual_array):
            n_penalized = len(x_actual_array) - len(x_actual_filtered)
            print(f"📊 Filtered out {n_penalized}/{len(x_actual_array)} penalized points, returning {len(x_actual_filtered)} for GP training")

        return x_actual_filtered, y_actual_filtered

    def _get_average_noise(self):
        """Calculate average noise estimate based on typical experimental variation."""
        if len(self.X_all_actual) >= 2:
            # Estimate noise based on inter-point distances in recent data
            # Use the standard deviation of distances between recent points as a proxy for noise
            recent_points = self.X_all_actual[-min(20, len(self.X_all_actual)):]
            if len(recent_points) >= 2:
                # Calculate pairwise distances and use their standard deviation as noise estimate
                distances = pdist(recent_points)
                noise_estimate = np.std(distances) * 0.1  # Scale down as proxy for coordinate noise
                return max(noise_estimate, 0.001)
            else:
                return 0.01
        else:
            # Default noise estimate when insufficient data
            return 0.01



    def update_penalization_mask(self):
        """Update penalization mask for all existing points when new needles are found."""
        if len(self.needles) == 0:
            return

        # Check all existing points against current needle locations
        if len(self.X_all_actual) > 0:
            # Only check points that weren't already penalized
            non_penalized_mask = ~self.all_penalized
            if np.any(non_penalized_mask):
                # Get all non-penalized points at once
                non_penalized_points = self.X_all_actual[non_penalized_mask]

                # Compute penalty mask for all non-penalized points in one call
                penalty_mask = self._compute_penalized_mask(non_penalized_points)

                # Update penalization status for points that are now in penalty regions
                newly_penalized = penalty_mask < 0.5
                self.all_penalized[non_penalized_mask] = newly_penalized

    def _compute_penalized_mask(self, mesh_points):
        """
        Compute penalized mask for given mesh points using current needles and penalty regions.

        Args:
            mesh_points: numpy array of shape (n_points, n_dimensions)

        Returns:
            numpy array of shape (n_points,) with mask (0 = penalized, 1 = allowed)
        """
        if len(self.needles) == 0:
            return np.ones(len(mesh_points))

        needle_locations = self.needles
        penalty_radii = self.penalty_regions
        n_points = len(mesh_points)

        # Convert needles to torch tensors for efficient computation
        needles_tensor = torch.tensor(
            np.array(needle_locations),
            dtype=torch.float32,
            device=self.device
        )  # Shape: (n_needles, n_dimensions)

        radii_tensor = torch.tensor(
            penalty_radii,
            dtype=torch.float32,
            device=self.device
        )  # Shape: (n_needles,)

        # Calculate optimal chunk size based on available GPU memory
        # Calculate chunk size based on point dimensions with safety multiplier
        point_dimensions = mesh_points.shape[-1]
        n_needles = len(self.needles)

        # Estimate memory per point: coordinates + distances to needles + intermediate tensors
        # Each point needs: dims (float32) + n_needles distances (float32) + intermediate tensors
        estimated_bytes_per_point = point_dimensions * 4 * 4  # 4 bytes per float32, 4x multiplier for intermediate tensors

        # Use available GPU memory with safety factor
        chunk_size = self.max_chunk_size  # Use parameter instead of hardcoded value
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = gpu_memory - torch.cuda.memory_allocated(self.device)

            # Use configurable percentage of available memory for safety
            safe_memory = available_memory * self.max_gpu_percentage

            # Calculate optimal chunk size based on memory per point
            optimal_chunk_size = int(safe_memory / estimated_bytes_per_point)

            # Cap at reasonable limits
            chunk_size = min(optimal_chunk_size, self.max_chunk_size)  # Max from parameter
            chunk_size = max(chunk_size, 1000)  # Min 1000 points

            if n_points > 1000:
                print(f"🚀 Computing penalty mask for {n_points:,} points using chunked GPU computation")
                print(f"   Points: {point_dimensions}D, Needles: {n_needles}, Memory/point: {estimated_bytes_per_point} bytes")
                print(f"   GPU memory: {gpu_memory/1024**3:.1f}GB, available: {available_memory/1024**3:.1f}GB")
                print(f"   Using chunk_size: {chunk_size:,}")

        penalty_mask = np.ones(n_points, dtype=np.float32)

        # Process points in chunks with automatic memory management
        with torch.no_grad():
            i = 0
            while i < n_points:
                end_idx = min(i + chunk_size, n_points)

                try:
                    # Convert chunk to torch tensor
                    batch_points = mesh_points[i:end_idx]
                    points_tensor = torch.tensor(
                        batch_points,
                        dtype=torch.float32,
                        device=self.device
                    )  # Shape: (chunk_size, n_dimensions)

                    # Compute distances from all points in chunk to all needles
                    # Expand dimensions for broadcasting
                    points_expanded = points_tensor.unsqueeze(1)  # (chunk_size, 1, n_dims)
                    needles_expanded = needles_tensor.unsqueeze(0)  # (1, n_needles, n_dims)

                    # Compute squared distances
                    diff = points_expanded - needles_expanded  # (chunk_size, n_needles, n_dims)
                    squared_distances = torch.sum(diff ** 2, dim=2)  # (chunk_size, n_needles)
                    distances = torch.sqrt(squared_distances)

                    # Check which points are within penalty radius of any needle
                    within_penalty = distances <= radii_tensor.unsqueeze(0)  # (chunk_size, n_needles)

                    # A point is penalized if it's within penalty radius of ANY needle
                    is_penalized = torch.any(within_penalty, dim=1)  # (chunk_size,)

                    # Convert back to numpy and update mask (0 = penalized, 1 = allowed)
                    batch_mask = (~is_penalized).float().cpu().numpy()
                    penalty_mask[i:end_idx] = batch_mask

                    # Clean up GPU memory after each chunk
                    del points_tensor, points_expanded, needles_expanded, diff, squared_distances, distances, within_penalty, is_penalized, batch_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Progress indicator for large datasets
                    if n_points > 1000000:
                        progress = end_idx / n_points * 100
                        if i % (chunk_size * 10) == 0:  # Print every 10 chunks
                            print(f"Progress: {progress:.1f}% ({end_idx:,}/{n_points:,})")

                    i = end_idx

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"GPU OOM at chunk {i}. Reducing chunk size and retrying...")
                        chunk_size = max(chunk_size // 2, 1000)
                        if chunk_size < 1000:
                            raise RuntimeError("Chunk size too small, cannot process dataset")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue  # Retry with smaller chunk size
                    else:
                        raise e

        return penalty_mask

    def calculate_penalty_radius_from_bounds_history(self, bounds_history):
        """Calculate penalty radius based on zoom level percentage using bounds history."""
        if not bounds_history or len(bounds_history) == 0:
            return 0.05  # Conservative default radius

        # Calculate the area at the specified zoom percentage
        total_zooms = len(bounds_history)
        target_zoom_level = max(1, round(self.penalty_zoom_percentage * total_zooms))

        if target_zoom_level <= len(bounds_history):
            target_bounds = bounds_history[target_zoom_level - 1]
            # Use average range across dimensions as radius
            ranges = [bounds[1] - bounds[0] for bounds in target_bounds]
            radius = np.mean(ranges) / 2.0
        else:
            # Fallback: use final zoom level
            final_bounds = bounds_history[-1]
            ranges = [bounds[1] - bounds[0] for bounds in final_bounds]
            radius = np.mean(ranges) / 2.0

        # Adaptive minimum radius based on search space
        # Use 1% of the total search space as absolute minimum
        total_space_range = np.mean([self.bounds[i][1] - self.bounds[i][0] for i in range(self.dimensions)])
        adaptive_min_radius = 0.01 * total_space_range

        # Add minimum radius based on 1.5x average noise (reduced from 3x)
        noise_based_min_radius = 1.5 * self._get_average_noise()

        # Final radius is max of calculated radius, adaptive minimum, and noise-based minimum
        final_radius = max(radius, adaptive_min_radius, noise_based_min_radius)

        # Cap maximum radius to prevent over-aggressive penalization
        # Use much smaller cap - 30% of total space or 0.3, whichever is smaller
        max_allowed_radius = min(0.3 * total_space_range, 0.3)
        final_radius = min(final_radius, max_allowed_radius)

        # Debug output
        print(f"🔍 Penalty radius calculation:")
        print(f"   Total zooms: {total_zooms}, Target zoom level: {target_zoom_level}")
        print(f"   Zoom percentage: {self.penalty_zoom_percentage}")
        print(f"   Calculated radius from zoom: {radius:.4f}")
        print(f"   Adaptive minimum: {adaptive_min_radius:.4f}")
        print(f"   Noise-based minimum (1.5x avg noise): {noise_based_min_radius:.4f}")
        print(f"   Max allowed: {max_allowed_radius:.4f}")
        print(f"   Final radius: {final_radius:.4f}")

        return final_radius

    def _get_unpenalized_points(self, limit_to_best=None):
        """Get unpenalized points from the global data arrays.

        Args:
            limit_to_best: If specified, keep only the best N points (smallest Y values)

        Returns:
            tuple: (X_actual, Y, mask) where mask is the boolean array
        """
        if len(self.Y_all) == 0:
            return (np.empty((0, self.dimensions)), np.empty(0), np.empty(0, dtype=bool))

        # Filter to non-penalized points only
        non_penalized_mask = ~self.all_penalized

        if np.sum(non_penalized_mask) == 0:
            return (np.empty((0, self.dimensions)), np.empty(0), non_penalized_mask)

        # Get non-penalized data
        filtered_X_actual = self.X_all_actual[non_penalized_mask]
        filtered_Y = self.Y_all[non_penalized_mask]

        # If limit_to_best is specified and we have more points than the limit
        if limit_to_best is not None and len(filtered_Y) > limit_to_best:
            # Get indices of best points (smallest Y values)
            best_indices = np.argsort(filtered_Y)[:limit_to_best]
            filtered_X_actual = filtered_X_actual[best_indices]
            filtered_Y = filtered_Y[best_indices]

        return filtered_X_actual, filtered_Y, non_penalized_mask

    # Methods moved from SimpleZoMBIZoom for integrated activation optimization
    def _initialize_activation_data(self):
        """Initialize activation-specific data storage with non-penalized data from global arrays."""
        # Get non-penalized data using helper method
        filtered_X_actual, filtered_Y, _ = self._get_unpenalized_points(limit_to_best=self.max_gp_points)

        if len(filtered_Y) > 0:
            self.activation_X_data = filtered_X_actual.copy()
            self.activation_Y_data = filtered_Y.copy()

            # Mark all initial points as zoom level -1
            self.zoom_level_markers = [-1] * len(self.activation_Y_data)
        else:
            # No non-penalized data available, initialize empty
            self.activation_X_data = np.empty((0, self.dimensions))
            self.activation_Y_data = np.empty(0)
            self.zoom_level_markers = []

        # Reset convergence tracking and bounds history for new activation
        self.convergence_errors = []
        self.bounds_history = []

    def _create_mesh(self, lower_bounds, upper_bounds):
        """Create a mesh of candidate points."""
        # Create grid for each dimension
        grids = []
        for i in range(self.dimensions):
            grids.append(np.linspace(lower_bounds[i], upper_bounds[i], self.resolution))

        # Create meshgrid
        mesh_grids = np.meshgrid(*grids)

        # Flatten and combine
        mesh_points = np.column_stack([grid.flatten() for grid in mesh_grids])
        return mesh_points

    def _gp_pred(self, X_candidates):
        """Predict GP model at given points with chunking for memory management."""
        n_candidates = len(X_candidates)

        # For small datasets, use direct prediction
        if n_candidates <= 1000:
            return self.gp.predict(X_candidates, return_std=True)

        # Calculate optimal chunk size based on available memory and model complexity
        point_dimensions = X_candidates.shape[-1]

        # Estimate memory per point for GP prediction
        # GP prediction involves kernel matrix computations which can be memory intensive
        # Base estimate: coordinates + kernel evaluations + gradients
        estimated_bytes_per_point = point_dimensions * 4 * 8  # 4 bytes per float32, 8x multiplier for GP operations

        # Default chunk size - use smaller default for GP operations
        chunk_size = self.max_chunk_size

        # Adjust chunk size based on available memory if using GPU
        if hasattr(self, 'device') and torch.cuda.is_available() and 'cuda' in str(self.device):
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = gpu_memory - torch.cuda.memory_allocated(self.device)

            # Use configurable percentage of available memory for GP predictions
            safe_memory = available_memory * self.max_gpu_percentage

            # Calculate optimal chunk size
            optimal_chunk_size = int(safe_memory / estimated_bytes_per_point)

            # Cap at reasonable limits
            chunk_size = min(optimal_chunk_size, self.max_chunk_size)  # Max from parameter
            chunk_size = max(chunk_size, 1000)  # Min 1K points

        # Log chunking for large datasets
        if n_candidates > 10000:
            n_chunks = (n_candidates + chunk_size - 1) // chunk_size
            print(f"🧠 GP prediction for {n_candidates:,} points using chunked computation")
            print(f"   Points: {point_dimensions}D, Memory/point: {estimated_bytes_per_point} bytes")
            print(f"   Using {n_chunks} chunks of size {chunk_size:,}")

        # Initialize result arrays
        all_means = []
        all_stds = []

        # Process points in chunks
        i = 0
        while i < n_candidates:
            end_idx = min(i + chunk_size, n_candidates)

            try:
                # Get chunk of candidates
                chunk_candidates = X_candidates[i:end_idx]

                # Make GP prediction on chunk
                chunk_means, chunk_stds = self.gp.predict(chunk_candidates, return_std=True)

                # Store results
                all_means.append(chunk_means)
                all_stds.append(chunk_stds)

                # Progress indicator for very large datasets
                if n_candidates > 100000:
                    progress = end_idx / n_candidates * 100
                    if i % (chunk_size * 10) == 0:  # Print every 10 chunks
                        print(f"GP Progress: {progress:.1f}% ({end_idx:,}/{n_candidates:,})")

                i = end_idx

            except (RuntimeError, MemoryError) as e:
                if 'out of memory' in str(e).lower() or 'memory' in str(e).lower():
                    print(f"GP OOM at chunk {i}. Reducing chunk size and retrying...")
                    chunk_size = max(chunk_size // 2, 500)
                    if chunk_size < 500:
                        raise RuntimeError("GP chunk size too small, cannot process dataset")

                    # Clear GPU memory if available
                    if hasattr(self, 'device') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue  # Retry with smaller chunk size
                else:
                    raise e

        # Concatenate all results
        final_means = np.concatenate(all_means)
        final_stds = np.concatenate(all_stds)

        return final_means, final_stds

    def _acquisition_function(self, X_candidates):
        """Expected Improvement acquisition function."""
        mu, sigma = self._gp_pred(X_candidates)

        # Find current best
        if len(self.activation_Y_data) == 0:
            return np.zeros(len(X_candidates))  # Return zeros if no data yet
        f_best = np.min(self.activation_Y_data)

        # Calculate Expected Improvement
        xi = 0.01  # exploration parameter
        with np.errstate(divide='warn'):
            Z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _penalty_acquisition_function(self, X_candidates):
        # Compute acquisition function
        base_acquisition_values = self._acquisition_function(X_candidates)

        # Apply penalty mask to acquisition values if we have previous needles
        if len(self.needles) > 0:
            penalty_mask = self._compute_penalized_mask(X_candidates)
            acquisition_values = base_acquisition_values.copy()
            # Compute the max absolute value for scaling the penalty
            max_abs_acq = np.max(np.abs(base_acquisition_values))
            penalty_value = -1000 * max_abs_acq if max_abs_acq > 0 else -1000.0
            acquisition_values[penalty_mask == 0] = penalty_value
        else:
            acquisition_values = base_acquisition_values

        return acquisition_values

    def _compute_new_bounds(self):
        """Compute new zoomed-in bounds based on top performing points."""
        # Get indices of top m points
        top_indices = np.argsort(self.activation_Y_data)[:self.top_m_points]
        top_points = self.activation_X_data[top_indices]

        # Compute bounds as min/max of top points with adaptive margin
        new_lower = []
        new_upper = []

        for dim in range(self.dimensions):
            dim_min = np.min(top_points[:, dim])
            dim_max = np.max(top_points[:, dim])
            dim_range = dim_max - dim_min

            # Adaptive margin based on current bounds size
            current_range = self.bounds[dim][1] - self.bounds[dim][0]
            base_margin = 0.15  # 15% base margin

            # Use base margin with minimum bound
            margin = max(base_margin * dim_range, 0.01 * current_range)

            # Add margin but keep within original bounds
            lower = max(self.bounds[dim][0], dim_min - margin)
            upper = min(self.bounds[dim][1], dim_max + margin)

            # Ensure minimum range to avoid overly tight bounds
            min_range = 0.01 * current_range
            if upper - lower < min_range:
                center = (upper + lower) / 2
                half_range = min_range / 2
                lower = max(self.bounds[dim][0], center - half_range)
                upper = min(self.bounds[dim][1], center + half_range)

            new_lower.append(lower)
            new_upper.append(upper)

        return new_lower, new_upper

    def _check_convergence(self, actual_y, predicted_y):
        """Check if we've converged to a needle."""
        if actual_y == 0:
            error = abs(predicted_y)
        else:
            error = abs(actual_y - predicted_y) / abs(actual_y)

        self.convergence_errors.append(error)

        # Check if last 3 errors are below tolerance
        if len(self.convergence_errors) >= 3:
            recent_errors = self.convergence_errors[-3:]
            if all(err <= self.tolerance for err in recent_errors):
                return True
        return False

    def _check_gp_size(self):
        """Check GP dataset size and trim to max_gp_points if needed, keeping best points."""
        if len(self.activation_Y_data) <= self.max_gp_points:
            return  # No trimming needed

        # Get indices of best (smallest Y values) points
        best_indices = np.argsort(self.activation_Y_data)[:self.max_gp_points]

        # Keep only the best points
        self.activation_X_data = self.activation_X_data[best_indices]
        self.activation_Y_data = self.activation_Y_data[best_indices]

        # Update zoom level markers
        self.zoom_level_markers = [self.zoom_level_markers[i] for i in best_indices]

        # Trim convergence errors to match (keep most recent)
        if len(self.convergence_errors) > self.max_gp_points:
            self.convergence_errors = self.convergence_errors[-self.max_gp_points:]

        print(f"🔄 GP dataset trimmed to {self.max_gp_points} best points")

    def run_activation(self, verbose=True):
        """
        Run a single activation (zoom sequence) to find one needle.

        Args:
            verbose: Print progress information

        Returns:
            dict: Results including best point, value, and convergence info
        """
        # Initialize activation data with non-penalized data from global arrays
        self._initialize_activation_data()

        if verbose:
            print(f"🔍 Initialized activation with {len(self.activation_Y_data)} points")
            if len(self.activation_Y_data) > 0:
                print(f"   Best initial value: {np.min(self.activation_Y_data):.6f}")

        current_bounds = [(bound[0], bound[1]) for bound in self.bounds]

        for zoom_level in range(self.max_zoom_levels):
            if verbose:
                print(f"\n=== Zoom Level {zoom_level + 1} ===")
                print(f"Current bounds: {current_bounds}")

            # Store bounds for this zoom level
            self.bounds_history.append([list(bound) for bound in current_bounds])

            # Create mesh for current bounds
            mesh_points = self._create_mesh([b[0] for b in current_bounds],
                                          [b[1] for b in current_bounds])

            # Check GP size and trim if needed before fitting
            self._check_gp_size()

            # Fit GP with current activation data
            if len(self.activation_Y_data) > 0:
                self.gp.fit(self.activation_X_data, self.activation_Y_data)

            iteration_converged = False

            for iteration in range(self.max_iterations):
                if verbose and iteration % 10 == 0:
                    best_y = np.min(self.activation_Y_data) if len(self.activation_Y_data) > 0 else float('inf')
                    print(f"  Iteration {iteration}, Best Y: {best_y:.6f}")

                # Compute acquisition function
                acquisition_values = self._penalty_acquisition_function(mesh_points)

                # Check if we have any valid acquisition values
                if np.all(acquisition_values == -np.inf):
                    if verbose:
                        print(f"⚠️  All acquisition values penalized, ending activation")
                        print(f"    Current penalty regions: {len(self.needles)} needles")
                        if len(self.needles) > 0:
                            print(f"    Penalty radii: {self.penalty_regions}")
                    break

                # Select best point to ask for
                best_idx = np.argmax(acquisition_values)
                x_ask = mesh_points[best_idx]

                # call linebo_sampler function - returns arrays of actual experimental points and values using linebo sampling
                x_actual_array, y_actual_array = self.linebo_sampler(x_ask)

                if y_actual_array.size == 0:          # everything was penalised
                    if verbose:
                        print("🛑 All sampled points were in penalty regions, "
                            "skipping this iteration.")
                    continue

                # Check convergence using the entire batch of experimental points
                # This is much more robust than using a single point
                converged = self._check_batch_convergence(x_actual_array, y_actual_array)

                # Find the best actual point from this batch for reporting
                best_batch_idx = np.argmin(y_actual_array)
                best_x_actual_batch = x_actual_array[best_batch_idx]
                best_y_actual = y_actual_array[best_batch_idx]

                # Add ALL experimental data from batch to activation training set
                for x_actual, y_actual in zip(x_actual_array, y_actual_array):
                    self.activation_X_data = np.vstack([self.activation_X_data, x_actual.reshape(1, -1)]) if len(self.activation_X_data) > 0 else x_actual.reshape(1, -1)
                    self.activation_Y_data = np.append(self.activation_Y_data, y_actual)
                    self.zoom_level_markers.append(zoom_level)

              # Check GP size and trim if needed before fitting
                self._check_gp_size()
                self.gp.fit(self.activation_X_data, self.activation_Y_data)

                if verbose and iteration % 10 == 0:
                    print(f"    Asked for: {x_ask}")
                    print(f"    Got {len(x_actual_array)} actual points, best Y: {best_y_actual:.6f}, "
                          f"Mean batch error: {self.convergence_errors[-1]:.6f}")

                if converged:
                    if verbose:
                        print(f"  *** NEEDLE FOUND at iteration {iteration}! ***")
                        print(f"  Convergence achieved with 3 consecutive errors < {self.tolerance}")
                        print(f"  Best actual point: {best_x_actual_batch}")
                        print(f"  Best Y value: {best_y_actual:.6f}")
                    iteration_converged = True
                    break

            if iteration_converged:
                break

            # Compute new bounds for next zoom level
            if zoom_level < self.max_zoom_levels - 1:  # Don't compute bounds on last iteration
                lower_bounds, upper_bounds = self._compute_new_bounds()
                current_bounds = list(zip(lower_bounds, upper_bounds))

                if verbose:
                    print(f"  Zooming in for next level...")

        # Find best result from activation
        best_idx = np.argmin(self.activation_Y_data)
        best_x_actual = self.activation_X_data[best_idx]
        best_y = self.activation_Y_data[best_idx]

        results = {
            'best_x_actual': best_x_actual,
            'best_y': best_y,
            'convergence_errors': self.convergence_errors.copy(),
            'total_evaluations': len(self.activation_Y_data),
            'all_x_actual': self.activation_X_data.copy(),
            'all_y': self.activation_Y_data.copy(),
            'converged': iteration_converged if 'iteration_converged' in locals() else False,
            'bounds_history': self.bounds_history.copy()
        }

        if verbose:
            print(f"\n=== ACTIVATION RESULTS ===")
            print(f"Best actual point: {best_x_actual}")
            print(f"Best value: {best_y:.6f}")
            print(f"Total evaluations: {len(self.activation_Y_data)}")
            print(f"Converged: {results['converged']}")

        return results

    def run_zombi_hop(self, verbose=True):
        """
        Main ZoMBI-Hop algorithm: multiple activations with penalization.

        Returns:
            dict: Results including all needles, values, and histories
        """
        if verbose:
            print(f"🚀 Starting ZoMBI-Hop with {self.num_activations} activations")
            print(f"📊 GP Memory limit: {self.max_gp_points} points")
            print(f"🎯 Penalty zoom percentage: {self.penalty_zoom_percentage*100:.0f}%")

        for activation in range(self.num_activations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔍 ACTIVATION {activation + 1}/{self.num_activations}")
                print(f"{'='*60}")

            # Run the activation directly using integrated method
            if verbose:
                print(f"🎯 Running zoom sequence...")
                if len(self.needles) > 0:
                    print(f"🚫 Penalty mask active based on {len(self.needles)} previous needle(s)")

            results = self.run_activation(verbose=verbose)

            # Store results
            if results['converged']:
                needle_point = results['best_x_actual']
                needle_value = results['best_y']

                # Calculate penalty radius for this needle using current bounds history
                penalty_radius = self.calculate_penalty_radius_from_bounds_history(results['bounds_history'])

                # Check for duplicate needles before adding
                is_duplicate = False
                if len(self.needles) > 0:
                    for existing_needle, existing_radius in zip(self.needles, self.penalty_regions):
                        distance = np.linalg.norm(needle_point - existing_needle)
                        # Use larger of the two radii for duplicate check
                        min_distance = max(existing_radius, penalty_radius)
                        if distance <= min_distance:
                            is_duplicate = True
                            if verbose:
                                print(f"⚠️  Duplicate needle detected (distance: {distance:.4f} <= {min_distance:.4f}) - ignoring")
                            break

                if not is_duplicate:
                    self.needles.append(needle_point)
                    self.needle_values.append(needle_value)
                    self.penalty_regions.append(penalty_radius)

                    # Update penalization mask for all existing points
                    self.update_penalization_mask()

                    if verbose:
                        print(f"✅ NEEDLE FOUND!")
                        print(f"   Location: {needle_point}")
                        print(f"   Value: {needle_value:.6f}")
                        print(f"   Penalty radius: {penalty_radius:.4f}")

                # Always append activation history (even for duplicates)
                self.activation_histories.append(results)

            else:
                if verbose:
                    print(f"❌ Activation {activation + 1} did not converge to a needle")
                self.activation_histories.append(results)

            if verbose:
                print(f"💾 Data collection status: {self.real_eval_count} real evaluations")
                if len(self.Y_all) > 0:
                    print(f"   Best value found: {np.min(self.Y_all):.6f}")
                    non_penalized_count = np.sum(~self.all_penalized)
                    print(f"   Non-penalized points: {non_penalized_count}/{len(self.Y_all)}")

            # No longer using surrogate model - all evaluations are real
            if verbose:
                non_penalized_count = np.sum(~self.all_penalized)
                penalized_count = np.sum(self.all_penalized)
                print(f"📊 Data distribution: {non_penalized_count} non-penalized, {penalized_count} penalized points")

        # Compile final results using proper evaluation counters
        total_evaluations = self.real_eval_count
        real_evaluations = self.real_eval_count
        surrogate_evaluations = 0

        results = {
            'needles': self.needles.copy(),
            'needle_values': self.needle_values.copy(),
            'penalty_regions': self.penalty_regions.copy(),
            'num_needles_found': len(self.needles),
            'total_evaluations': total_evaluations,
            'real_evaluations': real_evaluations,
            'surrogate_evaluations': surrogate_evaluations,
            'penalized_points': np.sum(self.all_penalized) if len(self.all_penalized) > 0 else 0,
            'best_value': np.min(self.Y_all) if len(self.Y_all) > 0 else None,
            'activation_histories': self.activation_histories,
            'converged_activations': sum([1 for hist in self.activation_histories
                                        if hist['converged']])
        }

        if verbose:
            print(f"\n🎉 ZoMBI-Hop Complete!")
            print(f"   Needles found: {results['num_needles_found']}/{self.num_activations}")
            print(f"   Total evaluations: {results['total_evaluations']}")
            print(f"   Real evaluations: {results['real_evaluations']}")
            print(f"   Surrogate evaluations: {results['surrogate_evaluations']}")
            print(f"   Penalized points: {results['penalized_points']}")
            print(f"   Experimental savings: {results['surrogate_evaluations']}/{results['total_evaluations']} ({100*results['surrogate_evaluations']/max(1,results['total_evaluations']):.1f}%)")
            print(f"   Best value found: {results['best_value']:.6f}" if results['best_value'] is not None else "None")

        return results



    def _linebo_project_simplex(self, v):
        """
        Project a point onto the unit simplex (sum=1, all components >= 0).

        Parameters:
        -----------
        v : array-like, shape (d,)
            Input vector to project

        Returns:
        --------
        array, shape (d,)
            Projected vector on the simplex
        """
        v = np.array(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v

        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, len(u) + 1)
        cond = u - cssv / ind > 0

        if np.any(cond):
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / float(rho)
        else:
            theta = 0

        w = np.maximum(v - theta, 0)
        return w

    def _linebo_generate_equally_spaced_directions(self, n_candidates, rng=None):
        """
        Directions that lie in the simplex hyperplane (sum=1) and are
        approximately uniformly spaced on the (d-2)-sphere in that subspace.
        """
        rng = np.random.default_rng(rng)
        d = self.dimensions

        # Build an orthonormal basis for the subspace {x : 1ᵀx = 0}
        # One easy way: Gram–Schmidt starting from the first d-1 standard
        # basis vectors minus the d-th.
        B = np.eye(d) - 1.0 / d
        Q, _ = np.linalg.qr(B.T)          # columns of Q span the subspace
        Q = Q[:, :-1]                     # remove the all-ones direction

        # Sample (or deterministically generate) directions on S^{d-2}
        # Here: random "spiral" method in d-1 dims, then push into full space
        dirs_sub = rng.normal(size=(n_candidates, d-1))
        dirs_sub /= np.linalg.norm(dirs_sub, axis=1, keepdims=True)

        directions = dirs_sub @ Q.T       # shape (n_candidates, d)
        return directions


    def _linebo_find_bounds_intersections(self, center_point, direction):
        """
        Find where a line intersects the bounds of the search space.

        For line: x(t) = center_point + t * direction
        Find t values where the line intersects self.bounds

        Parameters:
        -----------
        center_point : array-like, shape (d,)
            Starting point of line
        direction : array-like, shape (d,)
            Direction vector for the line

        Returns:
        --------
        tuple of (t_min, t_max)
            Parameter range where line intersects bounds
        """
        center_point = np.array(center_point)
        direction = np.array(direction)
        d = len(center_point)

        t_min = -np.inf
        t_max = np.inf

        for i in range(d):
            if abs(direction[i]) > 1e-10:  # Direction component is non-zero
                # Find t where line hits lower bound
                t_lower = (self.bounds[i][0] - center_point[i]) / direction[i]
                # Find t where line hits upper bound
                t_upper = (self.bounds[i][1] - center_point[i]) / direction[i]

                # Make sure t_lower <= t_upper
                if t_lower > t_upper:
                    t_lower, t_upper = t_upper, t_lower

                # Update global bounds
                t_min = max(t_min, t_lower)
                t_max = min(t_max, t_upper)

        # If no valid intersection, return center point
        if t_min >= t_max:
            return 0.0, 0.0

        return t_min, t_max

    def _linebo_find_line_boundaries(self, center_point, directions):
        """
        For each zero-sum direction v, find the largest t interval such that
        center_point + t v stays in the simplex (0 ≤ x_i ≤ 1).

        Parameters:
        -----------
        center_point : array-like, shape (d,)
            Starting point of lines
        directions : array, shape (n_lines, d)
            Direction vectors for each line (should be zero-sum)

        Returns:
        --------
        tuple of (t_start, t_end) arrays, each shape (n_lines,)
            Parameter ranges for valid line segments
        """
        c = np.asarray(center_point)
        v = np.asarray(directions)          # shape (n, d)

        # For every coordinate compute the t where it would hit 0 or 1
        with np.errstate(divide='ignore', invalid='ignore'):
            t_neg = np.where(v < -1e-12, -c        / v, -np.inf)  # toward 0
            t_pos = np.where(v >  1e-12, (1 - c)   / v,  np.inf)  # toward 1

        t_start = np.max(t_neg, axis=1)     # farthest we can go backward
        t_end   = np.min(t_pos, axis=1)     # farthest we can go forward

        # Collapse invalid lines to zero-length at the centre
        invalid = t_start >= t_end
        t_start[invalid] = 0.0
        t_end[invalid]   = 0.0
        return t_start, t_end

    def _linebo_integrate_acquisition_along_lines(self, center_point, directions, t_start, t_end,
                                                acquisition_fn, n_points_per_line=100):
        """
        Integrate acquisition function along each candidate line.

        Parameters:
        -----------
        center_point : array-like, shape (d,)
            Starting point of lines
        directions : array, shape (n_lines, d)
            Direction vectors
        t_start, t_end : arrays, shape (n_lines,)
            Parameter ranges for each line
        acquisition_fn : callable
            Function that takes X_candidates (shape n_points, d) and returns
            acquisition values (shape n_points,)
        n_points_per_line : int
            Number of points to sample along each line for integration

        Returns:
        --------
        array, shape (n_lines,)
            Integrated acquisition values for each line
        """
        center_point = np.array(center_point)
        n_lines = len(directions)
        integrals = np.zeros(n_lines)
        BAD = -10000  # replacement value

        for i in range(n_lines):
            if t_end[i] <= t_start[i]:
                integrals[i] = BAD  # Invalid line
                continue

            # Sample points along the line
            t_values = np.linspace(t_start[i], t_end[i], n_points_per_line)
            line_points = center_point + t_values[:, np.newaxis] * directions[i]

            # No need to project to simplex since directions are zero-sum
            # and we stay within bounds via t_start/t_end

            # Evaluate acquisition function
            acq_values = acquisition_fn(line_points)

            # Replace any -inf values with BAD
            acq_values = np.where(acq_values == -np.inf, BAD, acq_values)
            acq_values = np.where(np.isneginf(acq_values), BAD, acq_values)

            # Integrate (simple trapezoidal rule)
            dt = (t_end[i] - t_start[i]) / (n_points_per_line - 1) if n_points_per_line > 1 else 0
            integrals[i] = np.sum(acq_values) * dt

        # Replace any remaining -inf values with BAD
        integrals = np.where(integrals == -np.inf, BAD, integrals)
        integrals = np.where(np.isneginf(integrals), BAD, integrals)

        return integrals

    def linebo_sampler(self, x_tell):
        """
        LineBO sampler: takes a point (not necessarily in simplex), projects it to simplex,
        generates equally spaced lines through the projected point, evaluates penalty acquisition
        functions along each line, selects the best line, and returns the result of
        objective_function_wrapper called with the endpoints of the best line.

        Parameters:
        -----------
        x_tell : array-like, shape (d,)
            Input point (will be projected to simplex if needed)

        Returns:
        --------
        tuple of (x_sampled, y_measured)
            Result from objective_function_wrapper called with line endpoints
        """

        # Step 1: Project input point onto simplex
        x_tell_original = np.array(x_tell)
        x_tell = self._linebo_project_simplex(x_tell_original)

        print(f"LineBO input: {x_tell_original} -> projected to {x_tell}")
        print(f"Starting LineBO sampling from projected point: {x_tell}")

        # Step 2: Generate equally spaced line directions through the projected point
        print(f"Generating {self.linebo_num_lines} equally spaced line directions...")
        directions = self._linebo_generate_equally_spaced_directions(self.linebo_num_lines)

        # Step 3: Find valid parameter ranges for each line on the simplex
        print("Finding line boundary intersections with simplex...")
        t_start, t_end = self._linebo_find_line_boundaries(x_tell, directions)

        # Filter out invalid lines (where t_start >= t_end)
        valid_lines = t_start < t_end
        if not np.any(valid_lines):
            print("Warning: No valid lines found, using center point as both endpoints")
            endpoints = np.array([x_tell, x_tell])
            return self.objective_function_wrapper(endpoints)

        directions = directions[valid_lines]
        t_start = t_start[valid_lines]
        t_end = t_end[valid_lines]
        print(f"Found {len(directions)} valid lines")

        # Step 4: Integrate penalty acquisition function along each line
        print(f"Integrating penalty acquisition function along lines ({self.linebo_pts_per_line} points per line)...")
        integrals = self._linebo_integrate_acquisition_along_lines(
            x_tell, directions, t_start, t_end, self._penalty_acquisition_function, self.linebo_pts_per_line
        )

        # Step 5: Select the line with the best integrated acquisition (always maximize)
        best_line_idx = np.argmax(integrals)
        print(f"Selected line {best_line_idx} with max integrated acquisition: {integrals[best_line_idx]:.4f}")

        # Step 6: Find endpoints of the best line that intersect with bounds
        best_direction = directions[best_line_idx]

        # Find where this line intersects the search space bounds
        t_bounds_min, t_bounds_max = self._linebo_find_bounds_intersections(x_tell, best_direction)

        # Generate the two endpoints
        endpoint1 = x_tell + t_bounds_min * best_direction
        endpoint2 = x_tell + t_bounds_max * best_direction

        # Project endpoints to simplex to ensure constraints
        endpoint1 = self._linebo_project_simplex(endpoint1)
        endpoint2 = self._linebo_project_simplex(endpoint2)

        print(f"Line endpoints: {endpoint1} and {endpoint2}")

        # Step 7: Call objective_function_wrapper with the endpoints
        endpoints = np.array([endpoint1, endpoint2])
        print("Calling objective_function_wrapper with line endpoints...")

        return self.objective_function_wrapper(endpoints)

    def _check_convergence(self, actual_y, predicted_y):
        """
        DEPRECATED: Check if we've converged to a needle using single point.
        Use _check_batch_convergence() instead for better batch-aware convergence.
        """
        if actual_y == 0:
            error = abs(predicted_y)
        else:
            error = abs(actual_y - predicted_y) / abs(actual_y)

        self.convergence_errors.append(error)

        # Check if last 3 errors are below tolerance
        if len(self.convergence_errors) >= 3:
            recent_errors = self.convergence_errors[-3:]
            if all(err <= self.tolerance for err in recent_errors):
                return True
        return False

    def _check_batch_convergence(self, x_actual_array, y_actual_array):
        """
        Check convergence using the entire batch of experimental points.
        This is more robust than using a single point.

        Args:
            x_actual_array: Array of actual experimental points (n_experiments, dimensions)
            y_actual_array: Array of actual experimental values (n_experiments,)

        Returns:
            bool: True if converged
        """
        if len(x_actual_array) == 0:
            return False

        # Get GP predictions for all actual points in the batch
        y_pred_batch, y_std_batch = self.gp.predict(x_actual_array, return_std=True)

        # Calculate prediction errors for all points
        prediction_errors = []
        for y_actual, y_pred in zip(y_actual_array, y_pred_batch):
            if y_actual == 0:
                error = abs(y_pred)
            else:
                error = abs(y_actual - y_pred) / abs(y_actual)
            prediction_errors.append(error)

        prediction_errors = np.array(prediction_errors)

        # Calculate batch statistics
        mean_prediction_error = np.mean(prediction_errors)
        fraction_low_error = np.mean(prediction_errors <= self.tolerance)
        value_std = np.std(y_actual_array)
        best_value = np.min(y_actual_array)

        # Convergence criteria (multiple must be satisfied)
        criteria = {
            'mean_error_low': mean_prediction_error <= self.tolerance,
            'high_accuracy_fraction': fraction_low_error >= 0.6,  # 60% of points have low error
            'low_value_spread': value_std <= self.tolerance * best_value,  # Values are clustered
            'good_best_value': best_value <= np.percentile(y_actual_array, 25)  # Best value is actually good
        }

        # Store batch convergence metrics for tracking
        batch_metrics = {
            'mean_error': mean_prediction_error,
            'accuracy_fraction': fraction_low_error,
            'value_std': value_std,
            'best_value': best_value,
            'batch_size': len(y_actual_array)
        }

        # Store the mean error for tracking (backward compatibility)
        self.convergence_errors.append(mean_prediction_error)

        # Check if we have enough history
        if len(self.convergence_errors) < 3:
            return False

        # Must satisfy most criteria AND have consistent low errors over recent iterations
        criteria_met = sum(criteria.values()) >= 3  # At least 3 out of 4 criteria
        recent_errors_low = all(err <= self.tolerance * 1.5 for err in self.convergence_errors[-3:])  # Slightly more lenient for batch

        if criteria_met and recent_errors_low:
            return True

        return False


def general_line_objective(endpoint_batch, target_minima, n_experiments=24):
    """
    General objective function that takes endpoint pairs and returns experimental points
    along the line connecting them, with minima at specified target locations.

    This function is designed to work with the LineBO sampler which provides endpoints
    and expects a batch of experimental results along the line.

    Args:
        endpoint_batch: 2D numpy array of shape (2, dimensions) containing start and end points
        target_minima: List of target minimum locations (each should be on simplex)
        n_experiments: Number of experimental points to generate along the line

    Returns:
        x_actual_array: 2D numpy array of shape (n_experiments, dimensions)
        y_actual_array: 1D numpy array of shape (n_experiments,)
    """
    endpoint_batch = np.array(endpoint_batch)
    if endpoint_batch.shape[0] != 2:
        raise ValueError(f"Expected 2 endpoints, got {endpoint_batch.shape[0]}")

    start_point, end_point = endpoint_batch[0], endpoint_batch[1]
    dimensions = len(start_point)

    # Generate n_experiments points along the line between endpoints
    t_values = np.linspace(0, 1, n_experiments)
    line_points = []

    for t in t_values:
        # Linear interpolation between start and end
        point = (1 - t) * start_point + t * end_point
        line_points.append(point)

    line_points = np.array(line_points)

    # Add experimental noise to simulate real experimental variation
    noise_std = 0.015  # Slightly larger noise for realism
    x_actual_array = line_points + np.random.normal(0, noise_std, line_points.shape)

    # Ensure all points stay within [0,1] bounds
    x_actual_array = np.clip(x_actual_array, 0.0, 1.0)

    # Project to simplex to maintain constraints
    x_actual_array = np.array([project_to_simplex(point) for point in x_actual_array])

    # Evaluate objective function at each experimental point
    y_actual_array = []

    for x_actual in x_actual_array:
        # Compute distances to all target minima
        distances = [np.linalg.norm(x_actual - target) for target in target_minima]

        # Create objective value based on negative Gaussians at target points
        baseline = 1.0
        total_gaussian = 0.0

        for i, (target, distance) in enumerate(zip(target_minima, distances)):
            # Make first target global minimum, others local minima
            if i == 0:
                depth = -0.95  # Deeper well (global minimum)
                width = 12     # Well width parameter
            else:
                depth = -0.75  # Shallower well (local minimum)
                width = 12     # Same width

            total_gaussian += depth * np.exp(-width * distance**2)

        # Add gentle background variation
        background = 0.02 * np.sin(8 * np.sum(x_actual))

        # Add measurement noise
        measurement_noise = np.random.normal(0, 0.008)

        y_actual = baseline + total_gaussian + background + measurement_noise
        y_actual_array.append(y_actual)

    return x_actual_array, np.array(y_actual_array)


def general_single_point_objective(point, target_minima):
    """
    Single-point objective function that mimics the same landscape as general_line_objective.
    This is used for visualization and doesn't require line endpoints.

    Args:
        point: 1D numpy array of shape (dimensions,) - single point to evaluate
        target_minima: List of target minimum locations (each should be on simplex)

    Returns:
        float: Objective function value at the point
    """
    point = np.array(point)

    # Project to simplex to maintain constraints
    point = project_to_simplex(point)

    # Add small experimental noise for realism
    noise_std = 0.008
    measurement_noise = np.random.normal(0, noise_std)

    # Compute distances to all target minima
    distances = [np.linalg.norm(point - target) for target in target_minima]

    # Create objective value based on negative Gaussians at target points
    baseline = 1.0
    total_gaussian = 0.0

    for i, (target, distance) in enumerate(zip(target_minima, distances)):
        # Make first target global minimum, others local minima
        if i == 0:
            depth = -0.95  # Deeper well (global minimum)
            width = 12     # Well width parameter
        else:
            depth = -0.75  # Shallower well (local minimum)
            width = 12     # Same width

        total_gaussian += depth * np.exp(-width * distance**2)

    # Add gentle background variation
    background = 0.02 * np.sin(8 * np.sum(point))

    y_value = baseline + total_gaussian + background + measurement_noise
    return y_value


def plot_zombihop_results(optimizer_results, target_minima, show_penalty_regions=True, show_trajectories=True):
    """
    Comprehensive visualization of ZoMBI-Hop results using separate single-point objective function.

    Args:
        optimizer_results: Results dictionary from ZoMBI-Hop run
        target_minima: List of target minimum locations used to create the objective function
        show_penalty_regions: Whether to show penalty regions around needles
        show_trajectories: Whether to show optimization trajectories
    """
    # Extract data from optimizer results
    needles = optimizer_results.get('needles', [])
    needle_values = optimizer_results.get('needle_values', [])
    penalty_regions = optimizer_results.get('penalty_regions', [])
    activation_histories = optimizer_results.get('activation_histories', [])

    # Check if we have 2D data for visualization
    if len(target_minima) == 0 or len(target_minima[0]) != 2:
        print("Visualization only available for 2D problems")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define colors for different zoom levels
    activation_colors = plt.colormaps.get_cmap('Set1')(np.linspace(0, 1, len(activation_histories) + 1))

    # Plot 1: All trajectories and needles
    ax1 = axes[0, 0]

    for i, (history, color) in enumerate(zip(activation_histories, activation_colors)):
        if len(history['all_x_actual']) > 0:
            # Plot trajectory
            if show_trajectories:
                ax1.plot(history['all_x_actual'][:, 0], history['all_x_actual'][:, 1],
                       color=color, alpha=0.6, linewidth=2,
                       label=f'Activation {i+1} trajectory')

            # Plot sampled points
            ax1.scatter(history['all_x_actual'][:, 0], history['all_x_actual'][:, 1],
                       c=[color], alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

    # Plot needles
    if len(needles) > 0:
        needles_array = np.array(needles)
        for i, (needle, value, color) in enumerate(zip(needles_array, needle_values, activation_colors[:len(needles)])):
            ax1.scatter(needle[0], needle[1], c=[color], s=300, marker='*',
                       edgecolors='black', linewidth=2,
                       label=f'Needle {i+1} (y={value:.4f})')

    # Plot penalty regions
    if show_penalty_regions and len(needles) > 0:
        for needle, radius, color in zip(needles, penalty_regions, activation_colors[:len(needles)]):
            circle = plt.Circle(needle, radius, fill=False, edgecolor=color,
                              linestyle='--', linewidth=2, alpha=0.8)
            ax1.add_patch(circle)

    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title('ZoMBI-Hop: All Activations and Needles')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Plot 2: Target locations and discovered needles
    ax2 = axes[0, 1]

    # Plot target minima
    target_colors = ['red', 'orange', 'purple', 'brown', 'pink']
    for i, target in enumerate(target_minima[:5]):  # Limit to 5 targets for colors
        color = target_colors[i] if i < len(target_colors) else 'gray'
        ax2.scatter(target[0], target[1], c=color, s=200, marker='*',
                   edgecolors='black', linewidth=2, label=f'Target {i+1}', zorder=10)

    # Plot discovered needles
    if len(needles) > 0:
        needles_array = np.array(needles)
        for i, needle in enumerate(needles_array):
            ax2.scatter(needle[0], needle[1], c='lime', s=300, marker='X',
                       edgecolors='black', linewidth=2, label=f'Found {i+1}', zorder=10)

    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title('Targets vs Discovered Needles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Plot 3: True Objective Function Landscape
    ax3 = axes[1, 0]

    # Create a mesh for evaluating the true objective function
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)

    # Evaluate the true objective function on the mesh using single-point function
    Z_eval = np.zeros_like(X_eval)

    print(f"📊 Generating true function landscape ({len(x_eval)}×{len(y_eval)} grid)...")

    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            point = np.array([X_eval[i, j], Y_eval[i, j]])
            # Use single-point objective for clean evaluation
            Z_eval[i, j] = general_single_point_objective(point, target_minima)

    # Create smooth contour plot
    contour_levels = 20
    contour_lines = ax3.contour(X_eval, Y_eval, Z_eval, levels=contour_levels,
                               colors='black', linewidths=0.5, alpha=0.6)
    contour_filled = ax3.contourf(X_eval, Y_eval, Z_eval, levels=contour_levels,
                                 cmap='RdYlBu_r', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(contour_filled, ax=ax3, shrink=0.8)
    cbar.set_label('Objective Value', rotation=270, labelpad=15)

    # Add contour line labels
    ax3.clabel(contour_lines, inline=True, fontsize=7, fmt='%.2f',
              manual=False, inline_spacing=3)

    # Plot target minima
    for i, target in enumerate(target_minima[:5]):
        color = target_colors[i] if i < len(target_colors) else 'gray'
        ax3.scatter(target[0], target[1], c=color, s=200, marker='*',
                   edgecolors='black', linewidth=2, alpha=1.0, zorder=10)

    # Superimpose discovered needles
    if len(needles) > 0:
        needles_array = np.array(needles)
        for i, (needle, value) in enumerate(zip(needles_array, needle_values)):
            # Large X for needle
            ax3.scatter(needle[0], needle[1], c='lime', s=400, marker='X',
                       edgecolors='black', linewidth=2, alpha=1.0, zorder=10)

            # Add needle label with value
            ax3.annotate(f'N{i+1}\n{value:.3f}',
                       (needle[0], needle[1]),
                       xytext=(8, 8), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.9, edgecolor='black'),
                       fontsize=8, ha='left', va='bottom', weight='bold')

    # Add penalty regions as circles
    if show_penalty_regions and len(needles) > 0:
        for needle, radius in zip(needles, penalty_regions):
            circle = plt.Circle(needle, radius, fill=False, edgecolor='red',
                              linestyle='--', linewidth=2, alpha=0.7)
            ax3.add_patch(circle)

    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_title('True Objective Function Landscape')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sampled points and evaluation efficiency
    ax4 = axes[1, 1]

    # Get all sampled points from activation histories
    all_x_actual = []
    all_y_actual = []
    all_penalized = []

    for history in activation_histories:
        if len(history['all_x_actual']) > 0:
            all_x_actual.extend(history['all_x_actual'])
            all_y_actual.extend(history['all_y'])

    if len(all_x_actual) > 0:
        all_x_actual = np.array(all_x_actual)
        all_y_actual = np.array(all_y_actual)

        # Create a simple background landscape for context
        x_bg = np.linspace(0, 1, 30)
        y_bg = np.linspace(0, 1, 30)
        X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
        Z_bg = np.zeros_like(X_bg)

        for i in range(len(x_bg)):
            for j in range(len(y_bg)):
                point = np.array([X_bg[i, j], Y_bg[i, j]])
                Z_bg[i, j] = general_single_point_objective(point, target_minima)

        # Light background contours
        contour_bg = ax4.contourf(X_bg, Y_bg, Z_bg, levels=15, alpha=0.3, cmap='RdYlBu_r')

        # Plot all sampled points
        scatter = ax4.scatter(all_x_actual[:, 0], all_x_actual[:, 1],
                             c=all_y_actual, cmap='viridis', s=25, alpha=0.8,
                             edgecolors='black', linewidths=0.5, marker='o')

        plt.colorbar(scatter, ax=ax4, shrink=0.8, label='Objective Value')

        # Plot target minima
        for i, target in enumerate(target_minima[:5]):
            color = target_colors[i] if i < len(target_colors) else 'gray'
            ax4.scatter(target[0], target[1], c=color, s=200, marker='*',
                       edgecolors='black', linewidth=2, alpha=1.0, zorder=10)

        # Plot discovered needles
        if len(needles) > 0:
            needles_array = np.array(needles)
            for i, needle in enumerate(needles_array):
                ax4.scatter(needle[0], needle[1], c='lime', s=400, marker='X',
                           edgecolors='black', linewidth=2, alpha=1.0, zorder=10)

    ax4.set_xlabel('X1')
    ax4.set_ylabel('X2')
    ax4.set_title('All Sampled Points')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n📊 ZoMBI-Hop Visualization Summary:")
    print(f"   Targets: {len(target_minima)}")
    print(f"   Needles found: {len(needles)}")
    print(f"   Activations completed: {len(activation_histories)}")

    if len(needles) > 0:
        print(f"   Best needle value: {min(needle_values):.6f}")
        print(f"   Needle locations: {[(f'{n[0]:.3f}', f'{n[1]:.3f}') for n in needles]}")

        # Calculate distances to closest targets
        print(f"\n📏 Needle accuracy:")
        for i, needle in enumerate(needles):
            distances = [np.linalg.norm(needle - target) for target in target_minima]
            closest_idx = np.argmin(distances)
            distance = distances[closest_idx]
            print(f"   Needle {i+1}: distance to closest target = {distance:.4f}")


def test_general_zombihop(dimensions=2, n_experiments=24, num_activations=2,
                         min_distance=0.4, seed=42, verbose=True):
    """
    General testing function for ZoMBI-Hop that works in any dimension.

    Creates two random target minima on the simplex, tests the algorithm,
    and plots results if 2D.

    Args:
        dimensions: Number of dimensions to test
        n_experiments: Number of experiments per line evaluation
        num_activations: Number of activations (needles to find)
        min_distance: Minimum distance between target minima
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        dict: Results from ZoMBI-Hop including needles found and performance metrics
    """
    np.random.seed(seed)

    if verbose:
        print(f"🧪 Testing ZoMBI-Hop in {dimensions}D with {n_experiments} experiments per line")

    # Generate two random target minima on simplex that are at least min_distance apart
    target_minima = []
    max_attempts = 1000

    if verbose:
        print(f"📍 Generating target minima with minimum distance constraint: {min_distance}")

    for i in range(2):
        attempts = 0
        while attempts < max_attempts:
            # Generate random point on simplex
            random_point = np.random.random(dimensions)
            target = project_to_simplex(random_point)

            # Check minimum distance constraint
            if i == 0:
                target_minima.append(target)
                if verbose:
                    print(f"   Target 1: {target} (sum: {np.sum(target):.6f})")
                break
            else:
                distance = np.linalg.norm(target - target_minima[0])
                if distance >= min_distance:
                    target_minima.append(target)
                    if verbose:
                        print(f"   Target 2: {target} (sum: {np.sum(target):.6f})")
                        print(f"   ✅ Distance constraint satisfied: {distance:.4f} >= {min_distance}")
                    break

            attempts += 1

        if attempts >= max_attempts:
            raise ValueError(f"Could not generate targets with minimum distance {min_distance} after {max_attempts} attempts")

    # Final validation
    final_distance = np.linalg.norm(target_minima[1] - target_minima[0])
    if final_distance < min_distance:
        raise ValueError(f"VALIDATION ERROR: Final distance {final_distance:.4f} < required {min_distance}")

    if verbose:
        print(f"📏 Final target separation: {final_distance:.4f} (required: >= {min_distance})")
        print(f"   Constraint satisfied: {'✅' if final_distance >= min_distance else '❌'}")

    # Test the objective function directly
    if verbose:
        print(f"\n📊 Testing objective function at target points:")
        for i, target in enumerate(target_minima):
            # Create a small line segment around the target
            direction = np.random.normal(0, 0.1, dimensions)
            direction = direction / np.linalg.norm(direction) * 0.05  # Small perturbation

            endpoint1 = project_to_simplex(target - direction)
            endpoint2 = project_to_simplex(target + direction)
            endpoints = np.array([endpoint1, endpoint2])

            x_actual_array, y_actual_array = general_line_objective(endpoints, target_minima, n_experiments)
            avg_value = np.mean(y_actual_array)
            print(f"   Target {i+1}: {len(x_actual_array)} experiments, avg value = {avg_value:.6f}")

    # Create initialization data using targeted simplex sampling
    n_init = max(30, 5 * dimensions)  # Scale with dimensionality

    if verbose:
        print(f"\n📊 Creating initialization data with {n_init} simplex points...")

    # Create targeted initialization around the minima
    X_init_actual = create_targeted_simplex_initialization(
        n_samples=n_init,
        dimensions=dimensions,
        target_points=target_minima,
        target_fraction=0.4,  # 40% near targets
        seed=seed
    )

    # Simulate experimental noise
    noise_std = 0.012
    X_init_actual = X_init_actual + np.random.normal(0, noise_std, X_init_actual.shape)
    X_init_actual = np.array([project_to_simplex(point) for point in X_init_actual])

    # Evaluate objective function for initialization
    # For initialization, we'll evaluate at individual points by creating tiny line segments
    Y_init = []
    for x_actual in X_init_actual:
        # Create a tiny line segment around the point
        tiny_direction = np.random.normal(0, 0.001, dimensions)
        endpoint1 = project_to_simplex(x_actual - tiny_direction)
        endpoint2 = project_to_simplex(x_actual + tiny_direction)
        endpoints = np.array([endpoint1, endpoint2])

        x_actual_array, y_actual_array = general_line_objective(endpoints, target_minima, 3)  # Just 3 points
        Y_init.append(np.mean(y_actual_array))  # Use mean of the small line

    Y_init = np.array(Y_init)

    if verbose:
        print(f"   Generated {len(X_init_actual)} initialization points")
        print(f"   Noise std: {noise_std}")
        print(f"   Applied noise standard deviation: {noise_std:.4f}")
        print(f"   Best initial value: {np.min(Y_init):.6f}")
        print(f"   Worst initial value: {np.max(Y_init):.6f}")

    # Create objective function wrapper that uses our general_line_objective
    def objective_function_wrapper(endpoint_batch):
        return general_line_objective(endpoint_batch, target_minima, n_experiments)

    # Configure ZoMBI-Hop parameters based on dimensionality
    max_iterations = max(40, 10 * dimensions)
    resolution = max(15, 10 + dimensions)
    max_gp_points = max(150, 50 * dimensions)

    if verbose:
        print(f"\n🚀 Running ZoMBI-Hop...")
        print(f"   Dimensions: {dimensions}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Resolution: {resolution}")
        print(f"   GP points: {max_gp_points}")

    # Run ZoMBI-Hop
    optimizer = ZoMBIHop(
        objective_function=objective_function_wrapper,
        dimensions=dimensions,
        X_init_actual=X_init_actual,
        Y_init=Y_init,
        num_activations=num_activations,
        max_gp_points=max_gp_points,
        penalty_zoom_percentage=0.6,
        max_iterations=max_iterations,
        tolerance=0.06,
        top_m_points=max(4, dimensions),
        resolution=resolution,
        max_zoom_levels=6,
        num_experiments=n_experiments,
        linebo_num_lines=max(30, 10 * dimensions),
        linebo_pts_per_line=max(50, 20 * dimensions),
    )

    results = optimizer.run_zombi_hop(verbose=verbose)

    if verbose:
        print(f"\n📊 Results Summary:")
        print(f"   Needles found: {results['num_needles_found']}/{num_activations}")
        print(f"   Total evaluations: {results['total_evaluations']}")
        print(f"   Real evaluations: {results['real_evaluations']}")
        print(f"   Best value found: {results['best_value']:.6f}")

    # Analyze found needles
    if results['needles']:
        print(f"\n📍 Found Needles vs Expected:")
        for i, found_needle in enumerate(results['needles']):
            # Check simplex constraints
            needle_sum = np.sum(found_needle)
            needle_positive = np.all(found_needle >= 0)

            # Find closest target
            distances = [np.linalg.norm(found_needle - target) for target in target_minima]
            closest_idx = np.argmin(distances)
            closest_target = target_minima[closest_idx]
            distance = distances[closest_idx]

            print(f"   Needle {i+1}: {found_needle}")
            print(f"             Sum: {needle_sum:.6f}, All positive: {needle_positive}")
            print(f"             Value: {results['needle_values'][i]:.6f}")
            print(f"             Closest to target {closest_idx+1}: {closest_target}")
            print(f"             Distance: {distance:.4f}")
            print(f"             Penalty radius: {results['penalty_regions'][i]:.4f}")

            # Success threshold scales with dimensionality
            success_threshold = 0.15 + 0.05 * max(0, dimensions - 2)
            if distance < success_threshold:
                print(f"             ✅ Successfully found target {closest_idx+1}")
            else:
                print(f"             ⚠️  Distance {distance:.4f} > threshold {success_threshold:.4f}")

    # Plot results if 2D
    if dimensions == 2 and verbose:
        print(f"\n📈 Showing 2D visualization...")
        plot_zombihop_results(results, target_minima)

        # Additional plot showing true landscape with found needles
        plt.figure(figsize=(10, 8))

        # Create fine mesh for visualization
        x_range = np.linspace(0, 1, 40)
        y_range = np.linspace(0, 1, 40)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)

        print(f"📊 Generating 2D landscape visualization...")
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                point = np.array([X[i,j], Y[i,j]])
                point = project_to_simplex(point)

                # Create tiny line for evaluation
                tiny_dir = np.array([0.001, -0.001])
                endpoint1 = project_to_simplex(point - tiny_dir)
                endpoint2 = project_to_simplex(point + tiny_dir)
                endpoints = np.array([endpoint1, endpoint2])

                _, y_vals = general_line_objective(endpoints, target_minima, 3)
                Z[i,j] = np.mean(y_vals)

        # Plot landscape
        contour = plt.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
        plt.colorbar(contour, label='Objective Value')

        # Plot target minima
        target_colors = ['red', 'orange']
        for i, target in enumerate(target_minima):
            plt.scatter(target[0], target[1], c=target_colors[i], s=200,
                       marker='*', edgecolors='black', linewidth=2,
                       label=f'Target {i+1}', zorder=10)

        # Plot found needles
        if results['needles']:
            for i, needle in enumerate(results['needles']):
                plt.scatter(needle[0], needle[1], c='lime', s=300,
                           marker='X', edgecolors='black', linewidth=2,
                           label=f'Found {i+1}', zorder=10)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f'True 2D Landscape with Targets and Found Needles\n({n_experiments} experiments per line)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Store target information in results for analysis
    results['target_minima'] = target_minima
    results['dimensions'] = dimensions
    results['n_experiments'] = n_experiments

    return results


def test_multidimensional_performance():
    """Test ZoMBI-Hop performance across different dimensions."""
    print("🔬 Testing ZoMBI-Hop performance across dimensions...")

    dimensions_to_test = [2, 3, 4, 5]
    results_summary = []

    for dim in dimensions_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {dim}D optimization")
        print(f"{'='*60}")

        try:
            results = test_general_zombihop(
                dimensions=dim,
                n_experiments=20,  # Fewer experiments for speed
                num_activations=2,
                min_distance=0.4,
                seed=42,
                verbose=True
            )

            success_rate = results['num_needles_found'] / 2.0
            efficiency = results['real_evaluations'] / results['total_evaluations']

            results_summary.append({
                'dimensions': dim,
                'needles_found': results['num_needles_found'],
                'success_rate': success_rate,
                'total_evaluations': results['total_evaluations'],
                'real_evaluations': results['real_evaluations'],
                'efficiency': efficiency,
                'best_value': results['best_value']
            })

        except Exception as e:
            print(f"❌ Error in {dim}D: {e}")
            results_summary.append({
                'dimensions': dim,
                'needles_found': 0,
                'success_rate': 0.0,
                'total_evaluations': 0,
                'real_evaluations': 0,
                'efficiency': 0.0,
                'best_value': None
            })

    # Summary table
    print(f"\n{'='*80}")
    print("🎯 MULTIDIMENSIONAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dim':<4} {'Needles':<8} {'Success':<8} {'Total':<8} {'Real':<8} {'Efficiency':<10} {'Best':<10}")
    print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for result in results_summary:
        dim = result['dimensions']
        needles = f"{result['needles_found']}/2"
        success = f"{result['success_rate']:.1%}"
        total = result['total_evaluations']
        real = result['real_evaluations']
        efficiency = f"{result['efficiency']:.1%}"
        best = f"{result['best_value']:.4f}" if result['best_value'] is not None else "N/A"

        print(f"{dim:<4} {needles:<8} {success:<8} {total:<8} {real:<8} {efficiency:<10} {best:<10}")

    return results_summary


def create_simplex_initialization(n_samples, dimensions, seed=42):
    """
    Create initialization dataset on the simplex using Latin Hypercube Sampling + projection.

    All points will satisfy: sum(x) = 1 and x >= 0 for all components.

    Args:
        n_samples: Number of samples to generate
        dimensions: Number of dimensions
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (n_samples, dimensions) with all points on simplex
    """
    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)

    # Generate samples in [0,1] hypercube
    samples = sampler.random(n_samples)

    # Project each sample onto the simplex
    simplex_samples = []
    for sample in samples:
        # Use the standalone simplex projection function
        projected = project_to_simplex(sample)
        simplex_samples.append(projected)

    return np.array(simplex_samples)


def create_targeted_simplex_initialization(n_samples, dimensions, target_points,
                                         target_fraction=0.3, seed=42):
    """
    Create initialization dataset on simplex combining LHS exploration with targeted sampling.

    Args:
        n_samples: Total number of samples
        dimensions: Number of dimensions
        target_points: List of target points to sample around (should already be on simplex)
        target_fraction: Fraction of samples to place near targets
        seed: Random seed

    Returns:
        numpy array of shape (n_samples, dimensions) with all points on simplex
    """
    np.random.seed(seed)

    # Calculate how many samples for each category
    n_targeted = int(n_samples * target_fraction)
    n_lhs = n_samples - n_targeted

    samples = []

    # Generate LHS samples for global exploration (project to simplex)
    if n_lhs > 0:
        lhs_samples = create_simplex_initialization(n_lhs, dimensions, seed)
        samples.extend(lhs_samples)

    # Generate targeted samples around each target point (ensure they stay on simplex)
    if n_targeted > 0 and len(target_points) > 0:
        samples_per_target = n_targeted // len(target_points)
        remainder = n_targeted % len(target_points)

        for i, target in enumerate(target_points):
            # Add one extra sample to first 'remainder' targets
            n_for_this_target = samples_per_target + (1 if i < remainder else 0)

            for _ in range(n_for_this_target):
                # Generate noise in simplex-constrained way
                # Start with target, add small perturbation, then project
                noise_std = 0.05  # Smaller noise for simplex
                noise = np.random.normal(0, noise_std, dimensions)
                sample = np.array(target) + noise

                # Project to simplex to ensure constraints
                sample = project_to_simplex(sample)
                samples.append(sample)

    return np.array(samples)


if __name__ == "__main__":
    print("🧪 General ZoMBI-Hop Testing Suite")
    print("=" * 60)

    # Test 2D first (with visualization)
    print("🚀 Testing 2D optimization with visualization...")
    results_2d = test_general_zombihop(
        dimensions=2,
        n_experiments=24,
        num_activations=2,
        min_distance=0.4,
        seed=42,
        verbose=True
    )

    input("\nPress Enter to continue with 3D test...")

    # Test 3D
    print("\n🚀 Testing 3D optimization...")
    results_3d = test_general_zombihop(
        dimensions=3,
        n_experiments=24,
        num_activations=2,
        min_distance=0.4,
        seed=42,
        verbose=True
    )

    input("\nPress Enter to continue with multidimensional performance test...")

    # Test multiple dimensions
    print("\n🔬 Running multidimensional performance comparison...")
    performance_results = test_multidimensional_performance()

    print(f"\n✨ Testing complete!")
    print(f"   2D: {results_2d['num_needles_found']}/2 needles found")
    print(f"   3D: {results_3d['num_needles_found']}/2 needles found")

    # Provide recommendations
    print(f"\n🎯 Key Features of General Testing:")
    print(f"   - Works in any dimension (tested 2D-5D)")
    print(f"   - Automatically generates random simplex minima")
    print(f"   - Uses LineBO line-based experimental design")
    print(f"   - Scales parameters with dimensionality")
    print(f"   - Provides 2D visualization when applicable")
    print(f"   - Each line evaluation generates {results_2d['n_experiments']} experimental points")