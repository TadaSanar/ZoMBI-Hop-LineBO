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
import math
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
                 top_m_points=4, num_samples=15000, max_zoom_levels=4, bounds=None,
                 penalty_batch_size=10000, max_chunk_size=10000000, max_gpu_percentage=0.3,
                 num_experiments=20, linebo_num_lines=50, linebo_pts_per_line=100,
                 num_points_converge=3, radius_multiplier=10):
        """
        Initialize ZoMBI-Hop: Multi-activation optimization with adaptive penalization.

        ZoMBI-Hop is a Bayesian optimization algorithm that finds multiple local minima (needles)
        in high-dimensional spaces using a combination of Gaussian Process regression, LineBO
        sampling, and adaptive penalization to avoid revisiting previously discovered regions.

        Args:
            objective_function (callable): Function to optimize (minimize). Must accept a 2D numpy array
                of shape (2, dimensions) containing line endpoints and return two arrays:
                - x_actual_array: 2D numpy array of shape (n_experiments, dimensions) with actual
                  experimental points sampled along the line
                - y_actual_array: 1D numpy array of shape (n_experiments,) with corresponding
                  objective function values
            dimensions (int): Number of input dimensions for the optimization problem
            X_init_actual (numpy.ndarray): Initial dataset of actual experimental points.
                Shape: (n_points, dimensions). These points should already be evaluated
                and represent the starting dataset for optimization.
            Y_init (numpy.ndarray): Initial dataset of objective function values corresponding
                to X_init_actual. Shape: (n_points,). Must have same number of points as X_init_actual.
            num_activations (int, optional): Number of activations (needles to find). Default: 3.
                Each activation runs a complete zoom sequence to find one local minimum.
            max_gp_points (int, optional): Maximum number of points to keep in GP memory.
                Default: 200. When exceeded, only the best points are retained to manage memory.
            penalty_zoom_percentage (float, optional): Percentage of zoom levels to use for
                penalty area calculation (0.0-1.0). Default: 0.8. Controls how aggressive
                the penalty regions are based on zoom level history.
            max_iterations (int, optional): Maximum iterations per zoom level. Default: 20.
                Each zoom level runs up to this many iterations before zooming in.
            tolerance (float, optional): Convergence tolerance for needle detection. Default: 0.02.
                When prediction errors fall below this threshold, a needle is considered found.
            top_m_points (int, optional): Number of best points to use for bound computation.
                Default: 4. Used to determine new zoom bounds based on top-performing points.
            num_samples (int, optional): Number of random simplex samples to generate per iteration.
                Default: 15000. These samples are used for acquisition function evaluation.
            max_zoom_levels (int, optional): Maximum zoom levels per activation. Default: 4.
                Each activation can zoom in up to this many times before stopping.
            bounds (list of tuples, optional): List of (min, max) tuples for each dimension.
                Default: None (uses [0,1] for all dimensions). Defines the search space bounds.
            penalty_batch_size (int, optional): Batch size for penalty mask computation.
                Default: 10000. Larger values use more GPU memory but may be faster.
            max_chunk_size (int, optional): Maximum chunk size for GPU processing operations.
                Default: 10M points. Used for memory management in large-scale computations.
            max_gpu_percentage (float, optional): Maximum percentage of GPU memory to use
                for chunked operations (0.0-1.0). Default: 0.3 (30% of available GPU memory).
            num_experiments (int, optional): Number of experimental points to sample along
                the best line per acquisition. Default: 20. Each LineBO evaluation generates
                this many experimental points.
            linebo_num_lines (int, optional): Number of candidate lines to evaluate in LineBO.
                Default: 50. More lines provide better exploration but increase computation time.
            linebo_pts_per_line (int, optional): Number of points to sample per line for
                acquisition integration. Default: 100. Used to integrate acquisition function
                along each candidate line.
            num_points_converge (int, optional): Number of best points to use for convergence
                check and penalty radius calculation. Default: 3. When this many best points
                all have prediction errors below tolerance, convergence is declared.
            radius_multiplier (float, optional): Multiplicative factor to scale penalty radius.
                Default: 10. Larger values create larger penalty regions around found needles.

        Class Attributes Initialized:
            - self.needles: List to store best points found in each activation
            - self.needle_values: List to store best values found in each activation
            - self.penalty_regions: List to store penalty radii around each needle
            - self.activation_histories: List to store detailed history for each activation
            - self.X_all_actual: Array to store all actual sampled points (real evaluations only)
            - self.Y_all: Array to store all objective function values (real evaluations only)
            - self.all_penalized: Boolean array indicating which points were in penalized regions
            - self.real_eval_count: Counter for real objective function evaluations
            - self.device: GPU device ('cuda' or 'cpu') for torch operations
            - self.gp: GPUExactGP instance for Gaussian Process regression
            - self.activation_X_data: Activation-specific experimental points
            - self.activation_Y_data: Activation-specific experimental results
            - self.convergence_errors: List to track convergence errors
            - self.bounds_history: List to track bounds at each zoom level
            - self.zoom_level_markers: List to track which points belong to which zoom level

        Performance Features:
            - Optimized penalty mask computation using torch tensors with GPU acceleration
            - Chunked processing for large point sets to manage memory usage
            - Efficient vectorized distance calculations using broadcasting
            - Adaptive memory management based on available GPU memory
            - LineBO integration for efficient line-based experimental design
            - CFS (Constrained Fisher Sampling) for bounded simplex sampling

        Raises:
            ValueError: If input arrays have incorrect shapes or dimensions don't match
            RuntimeError: If GPU initialization fails or required dependencies are missing

        Example:
            >>> # Initialize with 2D problem and custom bounds
            >>> optimizer = ZoMBIHop(
            ...     objective_function=my_objective,
            ...     dimensions=2,
            ...     X_init_actual=np.random.random((50, 2)),
            ...     Y_init=np.random.random(50),
            ...     num_activations=2,
            ...     bounds=[(0, 1), (0, 1)]
            ... )
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.num_activations = num_activations
        self.max_gp_points = max_gp_points
        self.penalty_zoom_percentage = penalty_zoom_percentage
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.top_m_points = top_m_points
        self.num_samples = num_samples
        self.max_zoom_levels = max_zoom_levels
        self.penalty_batch_size = penalty_batch_size
        self.max_chunk_size = max_chunk_size
        self.max_gpu_percentage = max_gpu_percentage
        self.num_experiments = num_experiments
        self.linebo_num_lines = linebo_num_lines
        self.linebo_pts_per_line = linebo_pts_per_line
        self.num_points_converge = num_points_converge
        self.radius_multiplier = radius_multiplier

        # Set bounds
        if bounds is None:
            self.bounds = [(0, 1) for _ in range(dimensions)]
        else:
            self.bounds = bounds

        # Initialize storage for hop results
        self.needles = []  # Best points found in each activation
        self.needle_values = []  # Best values found in each activation
        self.penalty_regions = []  # Penalty regions around each needle
        self.needle_discovery_indices = []  # Evaluation indices when needles were discovered
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
        if len(self.Y_all) > 0:
            print(f"   Best initial value: {np.min(self.Y_all):.6f}")
        else:
            print(f"   Best initial value: N/A (no initial data)")
        print(f"   Average noise estimate: {self._get_average_noise():.6f}")

    def objective_function_wrapper(self, line_endpoints):
        """
        LineBO-compatible wrapper that manages objective function calls and penalty filtering.

        This method serves as the interface between LineBO sampling and the main objective function.
        It handles the complete workflow of calling the objective function, tracking all experimental
        points (both penalized and non-penalized), and filtering results for GP training.

        Args:
            line_endpoints (numpy.ndarray): 2D array of shape (2, dimensions) containing line
                start and end points. These define the line segment along which experimental
                points will be sampled by the objective function.

        Returns:
            tuple: (x_actual_filtered, y_actual_filtered) where:
                - x_actual_filtered (numpy.ndarray): 2D array of shape (n_filtered, dimensions)
                  containing only the non-penalized experimental points
                - y_actual_filtered (numpy.ndarray): 1D array of shape (n_filtered,) containing
                  the corresponding objective function values for non-penalized points

        Process:
            1. Calls the main objective function with line endpoints
            2. Computes penalty mask for all returned experimental points
            3. Tracks ALL points (penalized and non-penalized) in global arrays
            4. Updates evaluation counters
            5. Filters out penalized points for return to GP training
            6. Provides logging information about filtering results

        Class Parameters Used:
            - self.objective_function: The main objective function to call
            - self._compute_penalized_mask: Method to compute penalty regions
            - self.X_all_actual: Global array to store all actual experimental points
            - self.Y_all: Global array to store all objective function values
            - self.all_penalized: Boolean array tracking penalization status
            - self.real_eval_count: Counter for real evaluations

        Notes:
            - All experimental points are tracked globally regardless of penalization status
            - Only non-penalized points are returned for GP training to avoid bias
            - The method handles cases where all points might be penalized
            - Provides informative logging about the filtering process
        """
        # Call objective function with line endpoints
        x_actual_array, y_actual_array = self.objective_function(line_endpoints)

        # Check which points are in penalized regions
        penalty_mask = self._compute_penalized_mask(x_actual_array)
        penalty_status = penalty_mask < 0.5  # True if penalized

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
        """
        Calculate average noise estimate based on typical experimental variation.

        This method estimates the noise level in the experimental data by analyzing the
        spatial distribution of recent experimental points. It uses the standard deviation
        of pairwise distances between points as a proxy for coordinate-level noise.

        Returns:
            float: Estimated noise standard deviation. Minimum value is 0.001 to ensure
                numerical stability in GP training and convergence calculations.

        Method:
            1. Takes the most recent 20 experimental points (or all if fewer available)
            2. Computes pairwise Euclidean distances between all points
            3. Calculates standard deviation of these distances
            4. Scales by 0.1 to convert from distance space to coordinate space
            5. Applies minimum threshold of 0.001 for numerical stability

        Class Parameters Used:
            - self.X_all_actual: Global array of all actual experimental points
            - self.dimensions: Number of dimensions (used implicitly in distance calculations)

        Notes:
            - Used for setting minimum penalty radius thresholds
            - Provides adaptive noise estimation based on actual experimental data
            - Ensures numerical stability in GP training and convergence checks
            - Default value of 0.01 used when insufficient data is available
        """
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
        """
        Update penalization mask for all existing points when new needles are found.

        This method is called whenever a new needle is discovered to retroactively
        update the penalization status of all previously sampled points. This ensures
        that the global tracking arrays accurately reflect which points were in
        penalized regions when they were originally sampled.

        Process:
            1. Checks if any needles exist (if not, no penalization needed)
            2. Identifies all non-penalized points in the global dataset
            3. Computes penalty mask for these points using current needle locations
            4. Updates the penalization status for points that are now in penalty regions
            5. Only processes points that weren't already marked as penalized

        Class Parameters Used:
            - self.needles: List of discovered needle locations
            - self.X_all_actual: Global array of all actual experimental points
            - self.all_penalized: Boolean array tracking penalization status
            - self._compute_penalized_mask: Method to compute penalty regions

        Notes:
            - Only updates points that weren't already penalized to avoid redundant computation
            - Uses the current state of all needles to determine penalization
            - Ensures consistency between needle discovery and point penalization tracking
            - Called automatically when new needles are added to the system
        """
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

        This method efficiently determines which points in a large mesh are within penalty
        regions around previously discovered needles. It uses GPU-accelerated tensor operations
        with chunked processing to handle large point sets while managing memory usage.

        Args:
            mesh_points (numpy.ndarray): Array of shape (n_points, n_dimensions) containing
                the points to check for penalization. Can be any size from small batches
                to large mesh grids with millions of points.

        Returns:
            numpy.ndarray: Array of shape (n_points,) with penalty mask values:
                - 0.0: Point is penalized (within penalty radius of any needle)
                - 1.0: Point is allowed (not within penalty radius of any needle)

        Algorithm:
            1. If no needles exist, returns all-ones mask (no penalization)
            2. Converts needles and radii to torch tensors for GPU computation
            3. Calculates optimal chunk size based on available GPU memory
            4. Processes points in chunks to manage memory usage:
               - Converts chunk to torch tensor
               - Computes distances from all points in chunk to all needles
               - Uses broadcasting for efficient vectorized distance calculation
               - Checks which points are within penalty radius of any needle
               - Converts results back to numpy and updates mask
            5. Handles GPU out-of-memory errors by reducing chunk size and retrying
            6. Provides progress indicators for large datasets

        Class Parameters Used:
            - self.needles: List of discovered needle locations
            - self.penalty_regions: List of penalty radii around each needle
            - self.device: GPU device for torch operations
            - self.max_chunk_size: Maximum chunk size for processing
            - self.max_gpu_percentage: Maximum GPU memory usage percentage

        Performance Features:
            - GPU-accelerated distance calculations using torch tensors
            - Chunked processing for memory management
            - Automatic fallback with reduced chunk size on OOM errors
            - Progress tracking for large datasets
            - Efficient broadcasting for vectorized operations

        Notes:
            - A point is penalized if it's within penalty radius of ANY needle
            - Uses squared Euclidean distance for computational efficiency
            - Automatically manages GPU memory to prevent out-of-memory errors
            - Provides detailed logging for large datasets and memory management
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
                    if n_points > 100000:
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
        """
        DEPRECATED: Use calculate_penalty_radius instead.
        Calculate penalty radius based on zoom level percentage using bounds history.

        This method determines the appropriate penalty radius for a newly discovered needle
        based on the zoom level history from the activation that found it. The radius is
        calculated from the bounds at a specific zoom level, providing adaptive penalization
        that scales with the search resolution.

        Args:
            bounds_history (list): List of bounds at each zoom level during the activation.
                Each element is a list of (min, max) tuples for each dimension.
                Format: [[(min1, max1), (min2, max2), ...], ...] for each zoom level.

        Returns:
            float: Calculated penalty radius with applied multiplier. The radius is:
                - Based on the average range at the target zoom level
                - Bounded by adaptive minimum and maximum thresholds
                - Scaled by the radius_multiplier parameter

        Algorithm:
            1. Determines target zoom level based on penalty_zoom_percentage
            2. Extracts bounds from the target zoom level
            3. Calculates average range across all dimensions
            4. Applies adaptive minimum radius (1% of total search space)
            5. Applies noise-based minimum radius (1.5x average noise)
            6. Caps maximum radius to prevent over-aggressive penalization
            7. Applies radius_multiplier for final scaling

        Class Parameters Used:
            - self.penalty_zoom_percentage: Percentage of zoom levels to use for radius calculation
            - self.radius_multiplier: Multiplicative factor to scale the final radius
            - self.bounds: Global search space bounds for adaptive minimum calculation
            - self.dimensions: Number of dimensions for range calculations
            - self._get_average_noise: Method to estimate experimental noise

        Notes:
            - Uses zoom level history to provide context-aware penalization
            - Adaptive thresholds prevent overly small or large penalty regions
            - Noise-based minimum ensures penalty regions are larger than experimental noise
            - Maximum cap prevents over-aggressive penalization that could block exploration
            - Provides detailed debug output showing calculation steps
        """
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

        # Apply radius multiplier
        final_radius_with_multiplier = final_radius * self.radius_multiplier

        # Debug output
        print(f"🔍 Penalty radius calculation:")
        print(f"   Total zooms: {total_zooms}, Target zoom level: {target_zoom_level}")
        print(f"   Zoom percentage: {self.penalty_zoom_percentage}")
        print(f"   Calculated radius from zoom: {radius:.4f}")
        print(f"   Adaptive minimum: {adaptive_min_radius:.4f}")
        print(f"   Noise-based minimum (1.5x avg noise): {noise_based_min_radius:.4f}")
        print(f"   Max allowed: {max_allowed_radius:.4f}")
        print(f"   Final radius: {final_radius:.4f}")
        print(f"   Radius multiplier: {self.radius_multiplier}")
        print(f"   Final radius with multiplier: {final_radius_with_multiplier:.4f}")

        return final_radius_with_multiplier

    def calculate_penalty_radius(self, x_actual_array, y_actual_array):
        """
        Calculate penalty radius based on the best num_points_converge points from convergence check.

        This method determines the appropriate penalty radius for a newly discovered needle
        based on the spatial distribution of the best experimental points that led to
        convergence. The radius is calculated from the average distance between these
        converged points, providing a data-driven approach to penalization.

        Args:
            x_actual_array (numpy.ndarray): Array of actual experimental points from the
                convergence batch. Shape: (n_experiments, dimensions). These are the
                experimental points that were evaluated during the convergence check.
            y_actual_array (numpy.ndarray): Array of actual experimental values corresponding
                to x_actual_array. Shape: (n_experiments,). Used to identify the best points.

        Returns:
            float: Calculated penalty radius with applied multiplier. The radius is:
                - Based on average distance between the best converged points
                - Bounded by adaptive minimum and maximum thresholds
                - Scaled by the radius_multiplier parameter

        Algorithm:
            1. Identifies the best num_points_converge points (lowest y values)
            2. Calculates pairwise distances between all best points
            3. Uses average distance as the base radius
            4. Applies adaptive minimum radius (1% of total search space)
            5. Applies noise-based minimum radius (1.5x average noise)
            6. Caps maximum radius to prevent over-aggressive penalization
            7. Applies radius_multiplier for final scaling

        Class Parameters Used:
            - self.num_points_converge: Number of best points to use for radius calculation
            - self.radius_multiplier: Multiplicative factor to scale the final radius
            - self.bounds: Global search space bounds for adaptive minimum calculation
            - self.dimensions: Number of dimensions for range calculations
            - self._get_average_noise: Method to estimate experimental noise

        Notes:
            - Uses actual converged points to provide data-driven penalization
            - Adaptive thresholds prevent overly small or large penalty regions
            - Noise-based minimum ensures penalty regions are larger than experimental noise
            - Maximum cap prevents over-aggressive penalization that could block exploration
            - Provides detailed debug output showing calculation steps
            - Handles edge cases where only one point is available
        """
        if len(x_actual_array) == 0:
            return 0.05  # Conservative default radius

        # Get the best num_points_converge points (lowest y values)
        n_points_to_check = min(self.num_points_converge, len(y_actual_array))
        best_indices = np.argsort(y_actual_array)[:n_points_to_check]

        best_x_actual = x_actual_array[best_indices]

        # Calculate pairwise distances between the best points
        if len(best_x_actual) == 1:
            # Single point - use a small default radius
            radius = 0.05
        else:
            # Calculate distances between all pairs of best points
            distances = []
            for i in range(len(best_x_actual)):
                for j in range(i+1, len(best_x_actual)):
                    dist = np.linalg.norm(best_x_actual[i] - best_x_actual[j])
                    distances.append(dist)

            if len(distances) == 0:
                radius = 0.05
            else:
                # Use the average distance between converged points as the base radius
                radius = np.mean(distances)

        # Adaptive minimum radius based on search space
        # Use 1% of the total search space as absolute minimum
        total_space_range = np.mean([self.bounds[i][1] - self.bounds[i][0] for i in range(self.dimensions)])
        adaptive_min_radius = 0.01 * total_space_range

        # Add minimum radius based on 1.5x average noise
        noise_based_min_radius = 1.5 * self._get_average_noise()

        # Final radius is max of calculated radius, adaptive minimum, and noise-based minimum
        final_radius = max(radius, adaptive_min_radius, noise_based_min_radius)

        # Apply radius multiplier
        final_radius_with_multiplier = final_radius * self.radius_multiplier

        # Cap maximum radius to prevent over-aggressive penalization
        # Use 30% of total space or 0.3, whichever is smaller
        max_allowed_radius = min(0.3 * total_space_range, 0.3)
        final_radius_with_multiplier = min(final_radius_with_multiplier, max_allowed_radius)

        # Debug output
        print(f"🔍 Penalty radius calculation from {n_points_to_check} best points:")
        print(f"   Calculated radius from point distances: {radius:.4f}")
        print(f"   Adaptive minimum: {adaptive_min_radius:.4f}")
        print(f"   Noise-based minimum (1.5x avg noise): {noise_based_min_radius:.4f}")
        print(f"   Max allowed: {max_allowed_radius:.4f}")
        print(f"   Final radius: {final_radius:.4f}")
        print(f"   Radius multiplier: {self.radius_multiplier}")
        print(f"   Final radius with multiplier: {final_radius_with_multiplier:.4f}")

        return final_radius_with_multiplier

    def _get_unpenalized_points(self, limit_to_best=None):
        """
        Get unpenalized points from the global data arrays with optional filtering.

        This method provides access to the non-penalized experimental data for use in
        GP training and other operations. It can optionally filter to only the best
        performing points to manage memory usage and focus on high-quality data.

        Args:
            limit_to_best (int, optional): If specified, keep only the best N points
                (smallest Y values). If None, returns all non-penalized points.
                Default: None.

        Returns:
            tuple: (X_actual, Y, mask) where:
                - X_actual (numpy.ndarray): 2D array of shape (n_points, dimensions)
                  containing the non-penalized experimental points
                - Y (numpy.ndarray): 1D array of shape (n_points,) containing the
                  corresponding objective function values
                - mask (numpy.ndarray): Boolean array of shape (n_total,) indicating
                  which points in the global dataset are non-penalized

        Process:
            1. Creates boolean mask for non-penalized points
            2. Filters global arrays to non-penalized points only
            3. Optionally sorts by Y values and keeps only the best N points
            4. Returns filtered data along with the original mask

        Class Parameters Used:
            - self.X_all_actual: Global array of all actual experimental points
            - self.Y_all: Global array of all objective function values
            - self.all_penalized: Boolean array tracking penalization status

        Notes:
            - Returns empty arrays if no non-penalized points exist
            - When limit_to_best is specified, only the best points are returned
            - The mask preserves the original indexing of the global arrays
            - Used primarily for GP training and activation initialization
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
        """
        Initialize activation-specific data storage with non-penalized data from global arrays.

        This method prepares the data structures for a new activation by loading the best
        non-penalized experimental points from the global dataset. It ensures that each
        activation starts with high-quality data while respecting penalty regions from
        previously discovered needles.

        Process:
            1. Calls _get_unpenalized_points to retrieve non-penalized data
            2. Limits data to max_gp_points to manage memory usage
            3. Copies data to activation-specific storage arrays
            4. Initializes zoom level markers for all points
            5. Resets convergence tracking and bounds history for new activation

        Class Parameters Used:
            - self._get_unpenalized_points: Method to retrieve filtered data
            - self.max_gp_points: Maximum number of points to keep in GP memory
            - self.activation_X_data: Activation-specific experimental points storage
            - self.activation_Y_data: Activation-specific experimental results storage
            - self.zoom_level_markers: List to track which points belong to which zoom level
            - self.convergence_errors: List to track convergence errors
            - self.bounds_history: List to track bounds at each zoom level

        Notes:
            - Only non-penalized points are used to avoid bias from penalty regions
            - Data is limited to max_gp_points to manage memory usage
            - All initial points are marked as zoom level -1 (pre-activation)
            - Convergence tracking and bounds history are reset for clean start
            - Called at the beginning of each activation in run_activation
        """
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
        """
        Create random simplex points within the given bounds using CFS algorithm.

        This method generates a set of random points that satisfy the simplex constraint
        (sum of all components equals 1) while staying within the specified bounds.
        It uses the Constrained Fisher Sampling (CFS) algorithm for efficient and
        uniform sampling of the bounded simplex.

        Args:
            lower_bounds (list or numpy.ndarray): Lower bounds for each dimension.
                Shape: (dimensions,). Each value represents the minimum allowed value
                for that dimension.
            upper_bounds (list or numpy.ndarray): Upper bounds for each dimension.
                Shape: (dimensions,). Each value represents the maximum allowed value
                for that dimension.

        Returns:
            numpy.ndarray: Array of shape (num_samples, dimensions) containing
                randomly sampled points that satisfy:
                - Simplex constraint: sum(x_i) = 1 for each point
                - Bounds constraint: lower_bounds[i] <= x_i <= upper_bounds[i] for all i
                - Non-negativity: x_i >= 0 for all i

        Algorithm:
            1. Converts bounds to torch tensors for GPU computation
            2. Calls sample_bounded_simplex_cfs for efficient sampling
            3. Validates that all samples satisfy constraints
            4. Provides detailed logging for large sample sets
            5. Reports any constraint violations for debugging

        Class Parameters Used:
            - self.num_samples: Number of samples to generate
            - self.device: GPU device for torch operations
            - self.sample_bounded_simplex_cfs: Method implementing CFS algorithm

        Validation:
            - Checks that all samples are within specified bounds
            - Verifies simplex constraint (sum = 1) with tolerance 1e-6
            - Reports violations for debugging purposes
            - Ensures non-negativity of all components

        Notes:
            - Uses GPU-accelerated CFS algorithm for efficient sampling
            - Provides progress indicators for large sample sets
            - Includes comprehensive validation and error reporting
            - Handles edge cases and constraint violations gracefully
            - Used for acquisition function evaluation in each iteration
        """
        # Convert bounds to torch tensors
        a = torch.tensor(lower_bounds, dtype=torch.float32, device=self.device)
        b = torch.tensor(upper_bounds, dtype=torch.float32, device=self.device)

        if self.num_samples > 1000:
            print(f"🔄 Generating {self.num_samples:,} simplex samples using CFS algorithm")
            print(f"   Bounds: {[(f'{a[i]:.3f}', f'{b[i]:.3f}') for i in range(len(a))]}")

        # Generate samples using CFS algorithm (batching handled internally)
        samples_tensor = self.sample_bounded_simplex_cfs(
                    a=a,
                    b=b,
                    S=1.0,  # Simplex sum constraint
            num_samples=self.num_samples,
            # CFS direct sampler (no MCMC required)
        )

        # Convert to numpy
        final_samples = samples_tensor.cpu().numpy()

        if self.num_samples > 1000:
            print(f"✅ Generated {len(final_samples):,} simplex samples")
            print(f"   Sample verification: sum={np.mean(np.sum(final_samples, axis=1)):.6f}, "
                  f"min={np.min(final_samples):.6f}, max={np.max(final_samples):.6f}")

        # Additional validation - check all samples are within bounds
        within_bounds = np.all((final_samples >= np.array(lower_bounds)) &
                              (final_samples <= np.array(upper_bounds)), axis=1)
        sum_constraints = np.abs(np.sum(final_samples, axis=1) - 1.0) < 1e-6

        if not np.all(within_bounds):
            n_violations = np.sum(~within_bounds)
            print(f"⚠️  Warning: {n_violations}/{len(final_samples)} samples violate bounds!")
            print(f"   Bounds: {list(zip(lower_bounds, upper_bounds))}")
            violating_samples = final_samples[~within_bounds][:3]  # Show first 3
            for i, sample in enumerate(violating_samples):
                print(f"   Sample {i}: {sample}")

        if not np.all(sum_constraints):
            n_violations = np.sum(~sum_constraints)
            print(f"⚠️  Warning: {n_violations}/{len(final_samples)} samples violate simplex constraint!")

        return final_samples

    def _subset_sums_and_signs(self, caps: torch.Tensor) -> tuple:
        """
        Compute subset sums and inclusion-exclusion signs for CFS algorithm.

        This method implements the inclusion-exclusion principle for the Constrained
        Fisher Sampling (CFS) algorithm. It computes all possible subset sums of the
        given caps tensor and their corresponding signs for the inclusion-exclusion
        calculation.

        Args:
            caps (torch.Tensor): 1D tensor of shape (m,) containing the capacity values
                for each dimension. These represent the upper bounds minus lower bounds
                for the constrained sampling problem.

        Returns:
            tuple: (subset_sums, signs) where:
                - subset_sums (torch.Tensor): 1D tensor of shape (2**m,) containing
                  the sum of caps for each possible subset. Index i corresponds to
                  the subset defined by the binary representation of i.
                - signs (torch.Tensor): 1D tensor of shape (2**m,) containing ±1 values
                  for inclusion-exclusion. sign[i] = (-1)^|subset_i| where |subset_i|
                  is the cardinality of the subset corresponding to index i.

        Algorithm:
            1. Uses dynamic programming to compute subset sums efficiently
            2. Iterates through all possible bitmasks from 1 to 2**m-1
            3. For each bitmask, computes the sum of caps for the corresponding subset
            4. Computes signs using bit counting: (-1)^(number_of_set_bits)
            5. Returns both tensors for use in polytope volume calculations

        Mathematical Background:
            - Used in inclusion-exclusion principle for computing volumes of polytopes
            - Each subset sum represents the sum of capacities for a subset of dimensions
            - The signs alternate based on subset cardinality for inclusion-exclusion
            - Enables efficient computation of complex geometric volumes

        Notes:
            - Time complexity: O(2^m) where m is the number of dimensions
            - Space complexity: O(2^m) for storing all subset sums and signs
            - Used as a helper method in the CFS sampling algorithm
            - Critical for computing polytope volumes in bounded simplex sampling
        """
        m = caps.numel()
        device, dtype = caps.device, caps.dtype
        subset_sums = torch.zeros(1 << m, dtype=dtype, device=device)
        for mask in range(1, 1 << m):  # DP peel lowbit
            lsb = mask & -mask
            j = (lsb.bit_length() - 1)
            subset_sums[mask] = subset_sums[mask ^ lsb] + caps[j]
        # signs:  +1 for even |J|, ‑1 for odd |J|
        signs = torch.tensor([1 - 2 * (mask.bit_count() & 1) for mask in range(1 << m)],
                             dtype=dtype, device=device)
        return subset_sums, signs

    def _polytope_volume(self, S: torch.Tensor, subset_sums: torch.Tensor,
                        signs: torch.Tensor, power: int, denom: int) -> torch.Tensor:
        """
        Compute polytope volume or antiderivative using inclusion-exclusion principle.

        This method implements the core mathematical computation for the Constrained
        Fisher Sampling (CFS) algorithm. It computes either the volume of a polytope
        slice or its antiderivative using the inclusion-exclusion principle with
        vectorized operations for efficiency.

        Args:
            S (torch.Tensor): 1D tensor of shape (B,) containing the remaining sum
                values for each sample in the batch. These represent the remaining
                capacity to be allocated across dimensions.
            subset_sums (torch.Tensor): 1D tensor of shape (2**m,) containing precomputed
                subset sums from _subset_sums_and_signs method.
            signs (torch.Tensor): 1D tensor of shape (2**m,) containing inclusion-exclusion
                signs from _subset_sums_and_signs method.
            power (int): Power for the polynomial calculation. Use m-1 for volume
                computation and m for antiderivative computation, where m is the
                number of remaining dimensions.
            denom (int): Denominator factorial. Use (m-1)! for volume computation
                and m! for antiderivative computation.

        Returns:
            torch.Tensor: 1D tensor of shape (B,) containing the computed volumes
                or antiderivatives for each sample in the batch.

        Algorithm:
            1. Computes shifted values: S - subset_sums for all combinations
            2. Clamps negative values to zero (geometric constraint)
            3. Raises positive values to the specified power
            4. Applies inclusion-exclusion signs
            5. Sums over all subsets and divides by factorial denominator
            6. Applies numerical stability measures (clamping, NaN handling)

        Mathematical Background:
            - Implements inclusion-exclusion principle for polytope volume computation
            - The result represents the volume of the intersection of the simplex
              with the hyperplane defined by the remaining sum constraint
            - Used in CFS algorithm for computing conditional probabilities
            - Enables efficient sampling from bounded simplex regions

        Numerical Stability:
            - Clamps input values to prevent overflow/underflow
            - Handles NaN and Inf values with fallback strategies
            - Limits power calculations to prevent numerical issues
            - Applies final clamping to ensure non-negative results

        Notes:
            - Critical for the CFS sampling algorithm performance
            - Uses vectorized operations for batch processing
            - Includes comprehensive numerical stability measures
            - Handles edge cases and numerical errors gracefully
        """
        # Input validation
        if torch.any(torch.isnan(S)) or torch.any(torch.isinf(S)):
            print("🐛 Warning: NaN/Inf in S input to _polytope_volume")
            S = torch.clamp(S, min=0.0, max=1e6)

        # Shape (B, 2**m)
        shifted = (S.unsqueeze(1) - subset_sums)
        positive = torch.clamp(shifted, min=0.0)

        # Numerical stability: clamp power to avoid overflow
        if power > 50:  # Very high powers can cause overflow
            positive = torch.clamp(positive, max=10.0)

        powered = positive.pow(power)

        # Check for numerical issues in intermediate results
        if torch.any(torch.isnan(powered)) or torch.any(torch.isinf(powered)):
            print(f"🐛 Warning: NaN/Inf in powered terms (power={power})")
            print(f"   positive range: [{torch.min(positive):.2e}, {torch.max(positive):.2e}]")
            powered = torch.nan_to_num(powered, nan=0.0, posinf=1e6, neginf=0.0)

        result = (signs * powered).sum(dim=1) / denom

        # Final safety check
        result = torch.clamp(result, min=0.0)  # Volume should be non-negative
        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            print(f"🐛 Warning: NaN/Inf in _polytope_volume result")
            result = torch.nan_to_num(result, nan=1e-15, posinf=1e6, neginf=1e-15)

        return result


    def sample_bounded_simplex_cfs(
        self,
        a,
        b,
        S: float = 1.0,
        num_samples: int = 1000,
        max_batch: int = None,
        debug: bool = False,
        **ignored,
    ) -> torch.Tensor:
        """
        Generate highly-parallel CFS samples from bounded simplex.

        This method implements the Constrained Fisher Sampling (CFS) algorithm to generate
        independent and identically distributed (i.i.d.) points uniformly from the bounded
        simplex defined by {x | a ≤ x ≤ b, Σx = S}. The algorithm uses analytic inclusion-exclusion
        calculations for efficient and accurate sampling.

        Args:
            a (torch.Tensor): 1D tensor of shape (d,) containing lower bounds for each dimension.
                Must satisfy a[i] ≤ b[i] for all i.
            b (torch.Tensor): 1D tensor of shape (d,) containing upper bounds for each dimension.
                Must satisfy a[i] ≤ b[i] for all i.
            S (float, optional): Required total sum for the simplex constraint. Default: 1.0.
                Must satisfy Σa ≤ S ≤ Σb for feasibility.
            num_samples (int, optional): Number of points to generate. Default: 1000.
                Can be very large (billions) due to efficient implementation.
            max_batch (int, optional): Maximum batch size for parallel processing.
                Default: None (auto-calculated based on GPU memory). Used to manage
                GPU memory usage during sampling.
            debug (bool, optional): Enable debug output for numerical issues. Default: False.
                Provides detailed logging for troubleshooting numerical problems.
            **ignored: Additional keyword arguments (ignored for API compatibility).

        Returns:
            torch.Tensor: 2D tensor of shape (num_samples, d) containing the generated
                samples. Each row satisfies the simplex constraint Σx = S and bounds
                a ≤ x ≤ b.

        Algorithm:
            1. Validates input constraints (bounds, feasibility, dimension limits)
            2. Calculates optimal batch size based on available GPU memory
            3. Pre-computes subset sums and signs for inclusion-exclusion
            4. Generates samples in batches using coordinate-wise sampling:
               - For each coordinate, computes feasible interval
               - Uses Newton iteration to solve for sampling values
               - Handles deterministic cases (zero-length intervals)
               - Applies numerical stability measures
            5. Ensures final coordinate satisfies simplex constraint exactly
            6. Shifts results back to original coordinate system

        Mathematical Background:
            - CFS algorithm provides uniform sampling from bounded simplex
            - Uses inclusion-exclusion principle for volume calculations
            - Implements coordinate-wise conditional sampling
            - Maintains simplex constraint Σx = S throughout sampling
            - Provides i.i.d. samples with theoretical guarantees

        Performance Features:
            - GPU-accelerated with automatic memory management
            - Chunked processing for large sample sets
            - Progress tracking for long-running operations
            - Automatic fallback strategies for numerical issues
            - Efficient vectorized operations

        Class Parameters Used:
            - self.device: GPU device for torch operations
            - self.max_gpu_percentage: Maximum GPU memory usage percentage

        Validation:
            - Checks dimension limits (≤ 20 for analytic variant)
            - Validates bound constraints (a ≤ b)
            - Ensures feasibility (Σa ≤ S ≤ Σb)
            - Verifies simplex constraint in output
            - Reports constraint violations for debugging

        Notes:
            - Supports dimensions up to 20 for analytic variant
            - Can generate billions of samples efficiently
            - Includes comprehensive error handling and debugging
            - Provides detailed progress indicators for large datasets
            - Used as the core sampling method in _create_mesh
        """
        import math
        import numpy as np
        import torch
        # Convert a and b to torch tensors if they are numpy arrays
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        a = a.to(device=self.device, dtype=torch.float64)
        b = b.to(device=self.device, dtype=torch.float64)
        d = a.numel()

        if d > 20:
            raise ValueError("Analytic CFS variant supports dimension ≤ 20")
        if not torch.all(b >= a):
            raise ValueError("Each upper bound must exceed the lower bound")

        S = torch.as_tensor(S, dtype=torch.float64, device=self.device)
        if not (a.sum() - 1e-12 <= S <= b.sum() + 1e-12):
            raise ValueError("Sum S outside feasible range")

        # Calculate optimal batch size if not provided
        if max_batch is None:
            if self.device == 'cuda':
                # GPU: Memory-constrained batching
                estimated_bytes_per_point = d * 8 * 32  # Conservative estimate
                gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
                available_memory = gpu_memory - torch.cuda.memory_allocated(self.device)
                safe_memory = available_memory * self.max_gpu_percentage
                max_batch = max(int(safe_memory / estimated_bytes_per_point), 1000)
            else:
                # CPU: Large batches
                max_batch = min(num_samples, 10_000_000)

        if num_samples > 10_000:
            print(f"🚀 Generating {num_samples:,} samples using highly-parallel CFS")
            print(f"   Max batch size: {max_batch}")
            print(f"   Device: {self.device}")

        caps_full = b - a           # (d,)
        S0 = S - a.sum()            # remaining sum after shift to 0‑based caps

        # Create RNG
        rng = torch.Generator(device=self.device)

        # Pre‑compute subset‑sums / signs for each suffix caps[k:]
        precomp = []
        for k in range(d - 1):  # last coordinate is deterministic
            subset_sums, signs = self._subset_sums_and_signs(caps_full[k+1:])
            precomp.append((subset_sums, signs))

        out = torch.empty((num_samples, d), dtype=torch.float64, device=self.device)
        written = 0

        while written < num_samples:
            B = min(max_batch, num_samples - written)

            # Running arrays for the *entire* batch
            S_rem = S0.expand(B).clone()      # (B,)
            caps_rem = caps_full.clone()      # (d,)
            y = torch.empty((B, d), dtype=torch.float64, device=self.device)

            # Iterate over coordinates – small loop (≤20)
            for k in range(d - 1):
                subset_sums, signs = precomp[k]
                m = d - k - 1
                denom_vol = math.factorial(m - 1)
                denom_int = math.factorial(m)

                # Feasible interval per sample
                sum_tail_caps = caps_full[k+1:].sum()  # Use caps_full, not caps_rem
                t_low = torch.clamp(S_rem - sum_tail_caps, min=0.0)
                t_high = torch.minimum(caps_full[k].expand_as(S_rem), S_rem)

                # Short‑circuit deterministic cases (interval of length 0)
                deterministic_mask = (t_high - t_low) < 1e-15
                yk = torch.zeros_like(S_rem)  # Initialize with zeros
                yk[deterministic_mask] = t_low[deterministic_mask]

                # Stochastic sub‑batch
                stochastic_mask = ~deterministic_mask
                n_stochastic = stochastic_mask.sum().item()

                if n_stochastic > 0:
                    # Gather vectors for active samples
                    S_todo = S_rem[stochastic_mask]
                    tl = t_low[stochastic_mask]
                    th = t_high[stochastic_mask]

                    # Debug: Check for problematic intervals
                    interval_sizes = th - tl
                    if debug and torch.any(interval_sizes <= 0):
                        print(f"🐛 Debug: Zero or negative intervals detected in coord {k}")
                        print(f"   Min interval: {torch.min(interval_sizes):.2e}")
                        print(f"   Max interval: {torch.max(interval_sizes):.2e}")

                    # Skip samples with very small intervals to avoid numerical issues
                    valid_mask = interval_sizes > 1e-12
                    if torch.sum(valid_mask) == 0:
                        # All intervals too small, use midpoint
                        yk[stochastic_mask] = (tl + th) / 2
                        continue

                    # Only process valid samples
                    S_todo = S_todo[valid_mask]
                    tl_valid = tl[valid_mask]
                    th_valid = th[valid_mask]

                    # Helper lambdas (vectorised) with enhanced stability
                    def _volume(t: torch.Tensor) -> torch.Tensor:
                        vol = self._polytope_volume(S_todo - t, subset_sums, signs, m - 1, denom_vol)
                        # Clamp volume to avoid negative values due to numerical errors
                        return torch.clamp(vol, min=1e-15)

                    def _cdf(t: torch.Tensor) -> torch.Tensor:
                        shifted = S_todo - t
                        I_high = self._polytope_volume(shifted, subset_sums, signs, m, denom_int)
                        shifted_low = S_todo - tl_valid
                        I_low = self._polytope_volume(shifted_low, subset_sums, signs, m, denom_int)

                        # Ensure I_low >= I_high to avoid negative CDFs
                        I_low = torch.clamp(I_low, min=0.0)
                        I_high = torch.clamp(I_high, min=0.0)
                        I_high = torch.minimum(I_high, I_low)

                        cdf_val = I_low - I_high
                        return torch.clamp(cdf_val, min=1e-15)

                    # Normalisation constant Z ≡ integral from tl to th
                    Z = _cdf(th_valid)

                    # Debug: Check Z values
                    if torch.any(Z <= 0) or torch.any(torch.isnan(Z)) or torch.any(torch.isinf(Z)):
                        if debug:
                            print(f"🐛 Debug: Invalid Z values in coord {k}")
                            print(f"   Z min: {torch.min(Z):.2e}, max: {torch.max(Z):.2e}")
                            print(f"   NaN count: {torch.sum(torch.isnan(Z))}, Inf count: {torch.sum(torch.isinf(Z))}")
                        # Use fallback for all invalid samples
                        Z = torch.clamp(Z, min=1e-15)

                    U = torch.rand(len(tl_valid), generator=rng, device=self.device, dtype=torch.float64)
                    target = U * Z

                    # Newton iterations with enhanced stability and debugging
                    t = tl_valid + U * (th_valid - tl_valid)

                    for iteration in range(12):  # Increased iterations for better convergence
                        f = _cdf(t) - target
                        fp = _volume(t)

                        # Enhanced safe division
                        fp_safe = torch.clamp(fp.abs(), min=1e-15)
                        fp_sign = torch.sign(fp)

                        # Check for problematic derivatives
                        if debug and torch.any(fp_safe < 1e-12):
                            print(f"🐛 Debug: Very small derivatives in coord {k}, iter {iteration}")
                            print(f"   fp min: {torch.min(fp):.2e}, max: {torch.max(fp):.2e}")

                        delta = f / (fp_safe * fp_sign)

                        # Limit step size to prevent wild jumps
                        step_limit = 0.1 * (th_valid - tl_valid)
                        delta = torch.clamp(delta, -step_limit, step_limit)

                        t_new = t - delta
                        t = torch.clamp(t_new, tl_valid, th_valid)

                        # Check for NaN/Inf in intermediate results
                        if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                            if debug:
                                print(f"🐛 Debug: NaN/Inf in Newton iteration {iteration}, coord {k}")
                                print(f"   delta range: [{torch.min(delta):.2e}, {torch.max(delta):.2e}]")
                                print(f"   f range: [{torch.min(f):.2e}, {torch.max(f):.2e}]")
                                print(f"   fp range: [{torch.min(fp):.2e}, {torch.max(fp):.2e}]")
                            # Force fallback
                            t = tl_valid + torch.rand_like(tl_valid) * (th_valid - tl_valid)
                            break

                        # Check for convergence with relaxed tolerance
                        if torch.all(torch.abs(f) < 1e-10):
                            break

                    # Assign results back, handling the valid_mask
                    if torch.sum(valid_mask) > 0:
                        yk_temp = torch.zeros_like(S_rem[stochastic_mask])
                        yk_temp[valid_mask] = t
                        yk_temp[~valid_mask] = (tl[~valid_mask] + th[~valid_mask]) / 2  # Midpoint for invalid
                        yk[stochastic_mask] = yk_temp
                    else:
                        yk[stochastic_mask] = (tl + th) / 2

                # Validate coordinate values
                if torch.any(torch.isnan(yk)) or torch.any(torch.isinf(yk)):
                    print(f"⚠️  NaN/Inf detected in coordinate {k}, using enhanced fallback sampling")

                    if debug:
                        # Debug information about the problematic values
                        nan_count = torch.sum(torch.isnan(yk))
                        inf_count = torch.sum(torch.isinf(yk))
                        print(f"   NaN count: {nan_count}, Inf count: {inf_count}")
                        print(f"   S_rem range: [{torch.min(S_rem):.4f}, {torch.max(S_rem):.4f}]")
                        print(f"   t_low range: [{torch.min(t_low):.4f}, {torch.max(t_low):.4f}]")
                        print(f"   t_high range: [{torch.min(t_high):.4f}, {torch.max(t_high):.4f}]")

                    # Enhanced fallback: Use Beta distribution for more principled sampling
                    # Beta(2,2) gives a bell-shaped distribution on [0,1], more realistic than uniform
                    alpha, beta = 2.0, 2.0
                    U_beta = torch.distributions.Beta(alpha, beta).sample((len(S_rem),)).to(device=self.device, dtype=torch.float64)

                    # Ensure feasible intervals
                    interval_size = torch.clamp(t_high - t_low, min=1e-15)
                    yk = t_low + U_beta * interval_size

                    # Final safety check and clamp to valid range
                    yk = torch.clamp(yk, min=t_low, max=t_high)

                    if debug:
                        print(f"   Fallback yk range: [{torch.min(yk):.4f}, {torch.max(yk):.4f}]")

                # Write and update running state
                y[:, k] = yk
                S_rem -= yk

                # Ensure S_rem stays non-negative
                S_rem = torch.clamp(S_rem, min=0.0)

            # Last coordinate deterministic
            y[:, -1] = torch.minimum(torch.maximum(S_rem, torch.zeros_like(S_rem)),
                                   caps_full[-1].expand_as(S_rem))

            out[written: written+B] = y + a  # shift back
            written += B

            # Progress for large datasets
            if num_samples > 100_000 and written % max(max_batch, 100_000) == 0:
                progress = written / num_samples * 100
                print(f"CFS Progress: {progress:.1f}% ({written:,}/{num_samples:,})")

        if num_samples > 10_000:
            print(f"✅ Generated {len(out):,} highly-parallel CFS samples")

        return out

    def _gp_pred(self, X_candidates):
        """
        Predict GP model at given points with chunking for memory management.

        This method performs Gaussian Process predictions on a set of candidate points
        using chunked processing to manage memory usage, especially for large datasets
        or when using GPU acceleration. It automatically adjusts chunk sizes based on
        available memory and provides progress tracking for long-running operations.

        Args:
            X_candidates (numpy.ndarray): 2D array of shape (n_candidates, dimensions)
                containing the points at which to make GP predictions. Can be any size
                from small batches to large mesh grids.

        Returns:
            tuple: (means, stds) where:
                - means (numpy.ndarray): 1D array of shape (n_candidates,) containing
                  the predicted mean values at each candidate point
                - stds (numpy.ndarray): 1D array of shape (n_candidates,) containing
                  the predicted standard deviations at each candidate point

        Algorithm:
            1. Determines optimal chunk size based on available GPU memory
            2. Processes candidate points in chunks to manage memory usage
            3. Makes GP predictions on each chunk using the trained model
            4. Concatenates results from all chunks
            5. Handles out-of-memory errors by reducing chunk size and retrying
            6. Provides progress indicators for large datasets

        Memory Management:
            - Estimates memory per point based on model complexity
            - Calculates optimal chunk size from available GPU memory
            - Uses configurable percentage of available memory
            - Applies minimum and maximum chunk size limits
            - Automatically reduces chunk size on memory errors

        Class Parameters Used:
            - self.gp: Trained GPUExactGP model for predictions
            - self.device: GPU device for torch operations
            - self.max_chunk_size: Maximum chunk size for processing
            - self.max_gpu_percentage: Maximum GPU memory usage percentage

        Performance Features:
            - GPU-accelerated predictions when available
            - Automatic memory management and chunking
            - Progress tracking for large datasets
            - Robust error handling with automatic retry
            - Efficient concatenation of chunk results

        Notes:
            - Used by acquisition functions for optimization decisions
            - Handles both small and very large candidate sets efficiently
            - Provides detailed logging for memory management
            - Includes comprehensive error handling and recovery
            - Critical for the performance of the optimization algorithm
        """
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
        """
        Compute Expected Improvement acquisition function for Bayesian optimization.

        This method implements the Expected Improvement (EI) acquisition function,
        which balances exploration and exploitation in Bayesian optimization. It
        measures the expected improvement over the current best observed value,
        taking into account both the predicted mean and uncertainty from the GP model.

        Args:
            X_candidates (numpy.ndarray): 2D array of shape (n_candidates, dimensions)
                containing the candidate points at which to evaluate the acquisition
                function.

        Returns:
            numpy.ndarray: 1D array of shape (n_candidates,) containing the Expected
                Improvement values for each candidate point. Higher values indicate
                more promising points for evaluation.

        Algorithm:
            1. Handles bootstrap case when no training data is available
            2. Makes GP predictions (mean and standard deviation) at candidate points
            3. Identifies the best observed value from training data
            4. Computes Expected Improvement using the EI formula:
               EI = (f_best - μ - ξ) * Φ(Z) + σ * φ(Z)
               where Z = (f_best - μ - ξ) / σ
            5. Handles edge cases (zero standard deviation) gracefully

        Mathematical Background:
            - Expected Improvement balances exploration vs exploitation
            - Uses cumulative and probability density functions of normal distribution
            - Incorporates exploration parameter ξ to encourage exploration
            - Higher uncertainty (σ) increases acquisition value for exploration
            - Higher predicted improvement (f_best - μ) increases acquisition value

        Class Parameters Used:
            - self.activation_Y_data: Training data for identifying best observed value
            - self._gp_pred: Method for making GP predictions
            - self.gp: Trained Gaussian Process model

        Bootstrap Behavior:
            - When no training data is available, uses random exploration
            - Generates exponential random values to encourage exploration
            - Ensures optimization can start even with minimal data
            - Uses fixed seed for reproducibility

        Notes:
            - Critical for guiding the optimization process
            - Handles numerical edge cases gracefully
            - Provides fallback strategy for bootstrap scenarios
            - Used by _penalty_acquisition_function for final acquisition values
            - Exploration parameter ξ = 0.01 balances exploration and exploitation
        """
        # Handle bootstrap case when no training data is available
        if len(self.activation_Y_data) == 0:
            print("⚠️  No GP training data available, using random exploration")
            # Use random exploration when no data yet - this prevents zero acquisition
            # Generate random values with some variance to encourage exploration
            np.random.seed(42)  # For reproducibility
            random_values = np.random.exponential(scale=1.0, size=len(X_candidates))
            return random_values

        # Normal case: use GP predictions
        mu, sigma = self._gp_pred(X_candidates)
        f_best = np.min(self.activation_Y_data)

        # Calculate Expected Improvement
        xi = 0.01  # exploration parameter
        with np.errstate(divide='warn'):
            Z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _penalty_acquisition_function(self, X_candidates):
        """
        Compute acquisition function with penalty regions applied.

        This method combines the base acquisition function with penalty regions
        around previously discovered needles. It ensures that the optimization
        avoids revisiting areas around known local minima by applying strong
        penalties to points within penalty regions.

        Args:
            X_candidates (numpy.ndarray): 2D array of shape (n_candidates, dimensions)
                containing the candidate points at which to evaluate the acquisition
                function.

        Returns:
            numpy.ndarray: 1D array of shape (n_candidates,) containing the acquisition
                values with penalties applied. Points in penalty regions receive
                strongly negative values to discourage selection.

        Algorithm:
            1. Computes base acquisition function values using _acquisition_function
            2. If no needles exist, returns base acquisition values unchanged
            3. Computes penalty mask using _compute_penalized_mask
            4. Applies strong penalties to points in penalized regions:
               - Calculates penalty value as -1000 * max_absolute_acquisition_value
               - Ensures penalty is much lower than any valid acquisition value
            5. Returns modified acquisition values

        Penalty Strategy:
            - Uses adaptive penalty scaling based on acquisition function range
            - Penalty value = -1000 * max_absolute_acquisition_value
            - Ensures penalized points are never selected as optimal
            - Maintains relative ordering of non-penalized points
            - Provides fallback penalty value when acquisition range is zero

        Class Parameters Used:
            - self.needles: List of discovered needle locations
            - self._acquisition_function: Base acquisition function
            - self._compute_penalized_mask: Method to compute penalty regions

        Notes:
            - Critical for avoiding revisiting known local minima
            - Ensures exploration of new regions of the search space
            - Maintains the relative quality assessment of non-penalized points
            - Used by LineBO sampler for line selection
            - Provides adaptive penalty scaling for robustness
        """
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
        """
        Compute new zoomed-in bounds based on top performing points.

        This method implements the zooming strategy by computing new, tighter bounds
        around the best performing points from the current activation. It ensures
        that subsequent zoom levels focus on the most promising regions of the
        search space while maintaining minimum bounds to prevent overly restrictive
        search areas.

        Returns:
            tuple: (new_lower, new_upper) where:
                - new_lower (list): List of lower bounds for each dimension
                - new_upper (list): List of upper bounds for each dimension

        Algorithm:
            1. Identifies the top_m_points best performing points (lowest Y values)
            2. For each dimension, computes min and max of the best points
            3. Calculates adaptive margin based on current bounds size:
               - Base margin: 15% of the range of best points
               - Minimum margin: 1% of current bounds range
               - Uses the larger of these two values
            4. Applies margins while respecting global bounds
            5. Ensures minimum range to prevent overly tight bounds:
               - Minimum range: 1% of current bounds range
               - Centers the range if it becomes too small

        Class Parameters Used:
            - self.top_m_points: Number of best points to use for bound computation
            - self.activation_Y_data: Training data for identifying best points
            - self.activation_X_data: Training data for computing point ranges
            - self.bounds: Global search space bounds for constraint checking
            - self.dimensions: Number of dimensions for iteration

        Zoom Strategy:
            - Focuses search on regions containing the best observed points
            - Maintains exploration through adaptive margins
            - Prevents overly restrictive bounds that could miss optima
            - Respects global bounds to stay within valid search space
            - Ensures minimum range for numerical stability

        Notes:
            - Called at the end of each zoom level to prepare for next level
            - Adaptive margins scale with the quality of discovered points
            - Minimum range prevents numerical issues in subsequent iterations
            - Global bounds ensure search stays within valid domain
            - Critical for the zooming strategy of the algorithm
        """
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
        """
        Check if we've converged to a needle using single point evaluation.

        This method implements the convergence criterion for detecting when the
        optimization has found a local minimum (needle). It compares the actual
        experimental value with the GP prediction and tracks the error over time
        to determine if convergence has been achieved.

        Args:
            actual_y (float): The actual experimental value observed at the point
            predicted_y (float): The GP model prediction for the same point

        Returns:
            bool: True if convergence is detected, False otherwise. Convergence
                is declared when the last 3 prediction errors are all below
                the tolerance threshold.

        Algorithm:
            1. Computes relative prediction error:
               - If actual_y == 0: error = abs(predicted_y)
               - Otherwise: error = abs(actual_y - predicted_y) / abs(actual_y)
            2. Appends error to convergence_errors list
            3. Checks if last 3 errors are all below tolerance threshold
            4. Returns True if convergence criterion is met

        Convergence Criterion:
            - Requires the last 3 consecutive prediction errors to be below tolerance
            - Uses relative error to handle different scales of objective values
            - Handles edge case where actual value is zero
            - Provides stability against noise by requiring multiple consecutive successes

        Class Parameters Used:
            - self.tolerance: Convergence tolerance threshold
            - self.convergence_errors: List tracking prediction errors over time

        Notes:
            - DEPRECATED: Use _check_batch_convergence() instead for better batch-aware convergence
            - This method is kept for backward compatibility
            - Batch convergence provides more robust convergence detection
            - Used in early versions of the algorithm
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

    def _check_gp_size(self):
        """
        Check GP dataset size and trim to max_gp_points if needed, keeping best points.

        This method manages the memory usage of the Gaussian Process by monitoring
        the size of the training dataset and trimming it when it exceeds the
        maximum allowed size. It ensures that only the best performing points
        are retained, maintaining the quality of the GP model while controlling
        memory usage.

        Process:
            1. Checks if current dataset size exceeds max_gp_points
            2. If trimming is needed:
               - Identifies the best points (lowest Y values)
               - Keeps only the top max_gp_points points
               - Updates all related data structures consistently
               - Trims convergence errors to match
            3. Provides logging information about the trimming operation

        Class Parameters Used:
            - self.max_gp_points: Maximum number of points to keep in GP memory
            - self.activation_Y_data: Training data for identifying best points
            - self.activation_X_data: Training data to be trimmed
            - self.zoom_level_markers: Zoom level tracking to be trimmed
            - self.convergence_errors: Convergence tracking to be trimmed

        Memory Management:
            - Prevents exponential growth of GP training data
            - Maintains model quality by keeping best points
            - Ensures consistent state across all data structures
            - Provides informative logging about trimming operations
            - Critical for long-running optimizations

        Notes:
            - Called before GP fitting to ensure memory efficiency
            - Only trims when necessary (dataset size > max_gp_points)
            - Maintains relative ordering of points by quality
            - Updates all related tracking arrays consistently
            - Essential for preventing memory issues in long optimizations
        """
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

        This method implements the core optimization loop for a single activation.
        It performs a sequence of zoom levels, each consisting of multiple iterations
        of acquisition function optimization and experimental evaluation. The goal
        is to find one local minimum (needle) through progressive refinement of
        the search space.

        Args:
            verbose (bool, optional): Whether to print detailed progress information.
                Default: True. Controls the level of logging output during execution.

        Returns:
            dict: Results dictionary containing:
                - best_x_actual: Best experimental point found during activation
                - best_y: Best objective function value found
                - convergence_errors: List of prediction errors over time
                - total_evaluations: Total number of experimental evaluations
                - all_x_actual: All experimental points evaluated during activation
                - all_y: All objective function values from activation
                - converged: Boolean indicating if convergence was achieved
                - bounds_history: List of bounds used at each zoom level
                - converged_x_actual: Array of converged points (if convergence achieved)
                - converged_y_actual: Array of converged values (if convergence achieved)

        Algorithm:
            1. Initializes activation data with non-penalized points from global arrays
            2. For each zoom level (up to max_zoom_levels):
               a. Stores current bounds in bounds_history
               b. Creates mesh of candidate points using CFS sampling
               c. Fits GP model with current activation data
               d. For each iteration (up to max_iterations):
                  - Computes acquisition function with penalty regions
                  - Selects best candidate point
                  - Uses LineBO sampler to evaluate experimental points
                  - Checks for convergence using batch evaluation
                  - Adds experimental data to activation training set
                  - Fits GP model with updated data
                  - Breaks if convergence is achieved
               e. Computes new bounds for next zoom level (if not last level)
            3. Returns comprehensive results from the activation

        Class Parameters Used:
            - self.max_zoom_levels: Maximum number of zoom levels per activation
            - self.max_iterations: Maximum iterations per zoom level
            - self.bounds: Global search space bounds
            - self._initialize_activation_data: Method to initialize activation data
            - self._create_mesh: Method to generate candidate points
            - self._check_gp_size: Method to manage GP memory
            - self.gp: GPUExactGP model for predictions
            - self._penalty_acquisition_function: Method to compute acquisition values
            - self.linebo_sampler: Method to perform LineBO sampling
            - self._check_batch_convergence: Method to check convergence
            - self._compute_new_bounds: Method to compute new zoom bounds

        Convergence Detection:
            - Uses batch convergence checking for robustness
            - Requires multiple best points to have low prediction errors
            - Provides detailed convergence information
            - Stores converged points for analysis

        Notes:
            - Each activation is independent and focuses on finding one needle
            - Penalty regions from previous activations are respected
            - Zoom levels progressively refine the search space
            - LineBO sampling provides efficient experimental design
            - Comprehensive logging provides detailed execution information
        """
        # Initialize activation data with non-penalized data from global arrays
        self._initialize_activation_data()

        if verbose:
            print(f"🔍 Initialized activation with {len(self.activation_Y_data)} points")
            if len(self.activation_Y_data) > 0:
                print(f"   Best initial value: {np.min(self.activation_Y_data):.6f}")

        current_bounds = [(bound[0], bound[1]) for bound in self.bounds]

        # Initialize converged points storage
        final_converged_x_actual = np.array([])
        final_converged_y_actual = np.array([])

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
                x_actual_array, y_actual_array = self.linebo_sampler(x_ask, current_bounds)

                # Check convergence using the entire batch of experimental points
                # This is much more robust than using a single point
                converged, converged_x_actual, converged_y_actual = self._check_batch_convergence(x_actual_array, y_actual_array)

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
                        print(f"  Convergence achieved with best {len(converged_x_actual)} points < {self.tolerance}")
                        print(f"  Best actual point: {best_x_actual_batch}")
                        print(f"  Best Y value: {best_y_actual:.6f}")
                    # Store the converged points
                    final_converged_x_actual = converged_x_actual
                    final_converged_y_actual = converged_y_actual
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
            'bounds_history': self.bounds_history.copy(),
            'converged_x_actual': final_converged_x_actual,
            'converged_y_actual': final_converged_y_actual
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

        This method orchestrates the complete ZoMBI-Hop optimization process,
        running multiple activations to find multiple local minima (needles)
        while using adaptive penalization to avoid revisiting previously
        discovered regions. It manages the global state and coordinates
        between activations.

        Args:
            verbose (bool, optional): Whether to print detailed progress information.
                Default: True. Controls the level of logging output during execution.

        Returns:
            dict: Results dictionary containing:
                - needles: List of all discovered needle locations
                - needle_values: List of objective function values at needles
                - penalty_regions: List of penalty radii around each needle
                - num_needles_found: Number of successfully discovered needles
                - total_evaluations: Total number of experimental evaluations
                - real_evaluations: Number of real objective function evaluations
                - surrogate_evaluations: Number of surrogate model evaluations (always 0)
                - penalized_points: Number of points in penalized regions
                - best_value: Best objective function value found overall
                - activation_histories: List of detailed results from each activation
                - converged_activations: Number of activations that achieved convergence

        Algorithm:
            1. Initializes global tracking arrays and counters
            2. For each activation (up to num_activations):
               a. Runs single activation using run_activation
               b. If convergence achieved:
                  - Extracts best point and value from activation results
                  - Calculates penalty radius using converged points
                  - Checks for duplicate needles before adding
                  - Adds needle to global tracking if not duplicate
                  - Updates penalization mask for all existing points
               c. Stores activation history regardless of convergence
               d. Provides progress reporting and statistics
            3. Compiles final results with comprehensive statistics

        Class Parameters Used:
            - self.num_activations: Number of activations to run
            - self.needles: Global list to store discovered needle locations
            - self.needle_values: Global list to store needle objective values
            - self.penalty_regions: Global list to store penalty radii
            - self.activation_histories: Global list to store activation results
            - self.run_activation: Method to run individual activations
            - self.calculate_penalty_radius: Method to compute penalty radii
            - self.update_penalization_mask: Method to update penalty regions
            - self.X_all_actual: Global array of all experimental points
            - self.Y_all: Global array of all objective function values
            - self.all_penalized: Global array tracking penalization status
            - self.real_eval_count: Counter for real evaluations

        Duplicate Detection:
            - Checks distance between new needle and existing needles
            - Uses larger of two penalty radii for duplicate threshold
            - Prevents adding needles that are too close to existing ones
            - Maintains diversity in discovered local minima

        Penalty Management:
            - Calculates adaptive penalty radius based on converged points
            - Updates global penalization mask when new needles are found
            - Ensures subsequent activations avoid previously discovered regions
            - Provides detailed logging about penalty region creation

        Notes:
            - Each activation is independent and can find different local minima
            - Penalty regions prevent revisiting known local minima
            - Comprehensive tracking provides detailed optimization history
            - Robust duplicate detection ensures diverse needle discovery
            - Detailed logging provides insights into optimization progress
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

                # Calculate penalty radius using the best num_points_converge points from the activation data
                # Get the best num_points_converge points from the activation data
                n_points_for_radius = min(self.num_points_converge, len(self.activation_Y_data))
                best_indices = np.argsort(self.activation_Y_data)[:n_points_for_radius]
                best_x_for_radius = self.activation_X_data[best_indices]
                best_y_for_radius = self.activation_Y_data[best_indices]

                penalty_radius = self.calculate_penalty_radius(best_x_for_radius, best_y_for_radius)

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

                    # Track when this needle was discovered (current evaluation count)
                    self.needle_discovery_indices.append(self.real_eval_count)

                    # Update penalization mask for all existing points
                    self.update_penalization_mask()

                    if verbose:
                        print(f"✅ NEEDLE FOUND!")
                        print(f"   Location: {needle_point}")
                        print(f"   Value: {needle_value:.6f}")
                        print(f"   Penalty radius: {penalty_radius:.4f}")
                        print(f"   Discovered at evaluation: {self.real_eval_count}")

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
            'needle_discovery_indices': self.needle_discovery_indices.copy(),
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





    def _linebo_generate_equally_spaced_directions(self, n_candidates, rng=None):
        """
        Generate zero-sum directions for LineBO sampling on simplex.

        This method generates a set of direction vectors that maintain the simplex
        constraint when added to points on the simplex. Each direction vector has
        a sum of zero, ensuring that when added to a point with sum=1, the result
        also has sum=1, preserving the simplex constraint throughout line sampling.

        Args:
            n_candidates (int): Number of direction vectors to generate. This
                determines the number of candidate lines to evaluate in LineBO.
            rng (numpy.random.Generator or int, optional): Random number generator
                or seed for reproducibility. Default: None (uses default generator).

        Returns:
            numpy.ndarray: 2D array of shape (n_candidates, dimensions) containing
                the generated direction vectors. Each row is a direction vector
                that satisfies:
                - Zero-sum constraint: sum(v) = 0
                - Unit norm constraint: ||v|| = 1
                - Can be added to simplex points while maintaining simplex constraint

        Algorithm:
            1. Generates random Gaussian samples in R^d
            2. Enforces zero-sum constraint by subtracting mean of each row
            3. Normalizes each vector to unit length
            4. Handles edge case of zero vectors (rare but possible)
            5. Validates that all vectors satisfy zero-sum constraint

        Mathematical Background:
            - Zero-sum directions preserve simplex constraint: if x is on simplex
              (sum(x) = 1) and v is zero-sum (sum(v) = 0), then sum(x + t*v) = 1
            - Unit normalization ensures consistent line lengths
            - Gaussian sampling provides good coverage of direction space
            - Zero-sum constraint is enforced by centering each vector

        Class Parameters Used:
            - self.dimensions: Number of dimensions for vector generation

        Validation:
            - Checks that all generated directions have zero sum (within tolerance)
            - Reports any violations for debugging
            - Provides informative logging about direction generation

        Notes:
            - Critical for LineBO sampling on simplex-constrained problems
            - Ensures all line evaluations maintain simplex constraint
            - Provides good exploration of the simplex space
            - Used by linebo_sampler for generating candidate lines
        """
        # 1) RNG setup
        if isinstance(rng, np.random.Generator):
            rng_gen = rng
        else:
            rng_gen = np.random.default_rng(rng)

        d = self.dimensions

        # 2) Draw Gaussian samples
        V = rng_gen.standard_normal(size=(n_candidates, d))

        # 3) Enforce zero-sum: subtract mean of each row
        V -= V.mean(axis=1, keepdims=True)

        # 4) Normalize to unit length
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        # Avoid division by zero by handling rare zero-vector case
        norms[norms == 0] = 1.0
        V /= norms

        # Debug: verify directions are zero-sum
        direction_sums = np.sum(V, axis=1)
        if np.any(np.abs(direction_sums) > 1e-10):
            print(f"⚠️ Direction generation issue: sums = {direction_sums[:5]}")
            print(f"  Max deviation from zero-sum: {np.max(np.abs(direction_sums))}")

        # For debugging, show first few directions
        if n_candidates > 0:
            print(f"📐 Generated {n_candidates} zero-sum directions for {d}D simplex")
            print(f"  First direction: {V[0]}, sum: {np.sum(V[0]):.10f}")
            if len(V) > 1:
                print(f"  Second direction: {V[1]}, sum: {np.sum(V[1]):.10f}")

        return V


    def _linebo_find_line_endpoints(self, x_tell, direction, bounds):
        """
        Find line endpoints where a line through x_tell intersects the bounds.

        This method computes the intersection points of a line with the search space
        bounds. The line is defined by a starting point x_tell and a direction vector.
        It finds the two points where this line intersects the boundary of the
        search space, creating a line segment for evaluation.

        Args:
            x_tell (numpy.ndarray): 1D array of shape (dimensions,) containing the
                starting point of the line. This is typically the current best point
                or a point of interest for exploration.
            direction (numpy.ndarray): 1D array of shape (dimensions,) containing the
                direction vector. This defines the direction of the line from x_tell.
            bounds (list of tuples): List of (min, max) tuples for each dimension,
                defining the search space bounds.

        Returns:
            tuple or None: If a valid line segment exists:
                - (start_point, end_point): Tuple of two 1D arrays, each of shape
                  (dimensions,), representing the intersection points with bounds
                - None: If no valid line segment exists (line doesn't intersect bounds
                  or intersection is empty)

        Algorithm:
            1. For each dimension, computes intersection parameters t where:
               - Lower bound: t_lower = (bounds[i][0] - x_tell[i]) / direction[i]
               - Upper bound: t_upper = (bounds[i][1] - x_tell[i]) / direction[i]
            2. Handles zero direction components (line parallel to coordinate axis)
            3. Finds the intersection interval [t_min, t_max] across all dimensions
            4. Checks if valid interval exists (t_min ≤ t_max)
            5. Computes endpoints: start_point = x_tell + t_min * direction
               end_point = x_tell + t_max * direction
            6. Clips endpoints to bounds to handle numerical errors

        Mathematical Background:
            - Line equation: x(t) = x_tell + t * direction
            - Intersection with bounds: bounds[i][0] ≤ x_tell[i] + t * direction[i] ≤ bounds[i][1]
            - Valid line segment requires non-empty intersection across all dimensions
            - Endpoints represent the boundary of the feasible line segment

        Edge Cases:
            - Handles zero direction components (vertical lines)
            - Checks if starting point is within bounds
            - Clips endpoints to handle numerical precision issues
            - Returns None for invalid or empty intersections

        Notes:
            - Critical for LineBO sampling to ensure valid line segments
            - Ensures all evaluated points are within search space bounds
            - Handles numerical precision issues gracefully
            - Used by linebo_sampler for line generation
        """
        # Find t values where line x_tell + t*direction hits each bound
        t_min = -np.inf
        t_max = np.inf

        for i in range(len(x_tell)):
            if direction[i] != 0:
                # Check lower bound
                t_lower = (bounds[i][0] - x_tell[i]) / direction[i]
                # Check upper bound
                t_upper = (bounds[i][1] - x_tell[i]) / direction[i]

                # Ensure t_lower <= t_upper
                if direction[i] < 0:
                    t_lower, t_upper = t_upper, t_lower

                t_min = max(t_min, t_lower)
                t_max = min(t_max, t_upper)
            else:
                # Direction is zero in this dimension, check if x_tell is within bounds
                if x_tell[i] < bounds[i][0] or x_tell[i] > bounds[i][1]:
                    return None

        # Check if valid line segment exists
        if t_min > t_max:
            return None

        # Calculate endpoints
        start_point = x_tell + t_min * direction
        end_point = x_tell + t_max * direction

        # Ensure endpoints are within bounds (handle numerical errors)
        start_point = np.clip(start_point, [b[0] for b in bounds], [b[1] for b in bounds])
        end_point = np.clip(end_point, [b[0] for b in bounds], [b[1] for b in bounds])

        return (start_point, end_point)

    def _linebo_integrate_acquisition(self, endpoints, num_points, penalty_acquisition):
        """
        Integrate acquisition function along a line segment.

        This method evaluates the acquisition function at multiple points along a
        line segment and computes the sum of these values. This integration is used
        in LineBO to compare different candidate lines and select the most promising
        one for experimental evaluation.

        Args:
            endpoints (tuple): Tuple of (start_point, end_point) where each is a
                1D array of shape (dimensions,) defining the line segment endpoints.
            num_points (int): Number of points to sample along the line for
                integration. More points provide more accurate integration but
                increase computation time.
            penalty_acquisition (callable): Acquisition function that takes a 2D
                array of points and returns acquisition values. Should handle
                infeasible points by returning np.inf.

        Returns:
            float: Integrated acquisition value along the line segment. This is
                the sum of acquisition values at evenly spaced points along the
                line, with infeasible points (np.inf) replaced by -10000.

        Algorithm:
            1. Generates evenly spaced points along the line segment using linear
               interpolation between endpoints
            2. Evaluates the acquisition function at all points simultaneously
            3. Replaces any np.inf values with -10000 (strong penalty for infeasible points)
            4. Returns the sum of all acquisition values

        Mathematical Background:
            - Line parameterization: x(t) = (1-t) * start_point + t * end_point
            - Integration: sum of acquisition values at evenly spaced t values
            - Provides measure of line quality for LineBO selection
            - Higher integrated values indicate more promising lines

        Class Parameters Used:
            - self.linebo_pts_per_line: Number of points for integration (if not specified)

        Integration Strategy:
            - Uses evenly spaced points for uniform sampling
            - Handles infeasible points with strong penalties
            - Provides scalar measure for line comparison
            - Used by LineBO for line selection

        Notes:
            - Critical for LineBO algorithm to compare candidate lines
            - Handles infeasible points gracefully with penalty values
            - Provides scalar measure for optimization over line space
            - Used by linebo_sampler for line evaluation
        """
        if self.gp is None:
            print("GP is not initialized. Returning random value between 0 and 1.")
            return np.random.rand()
        start_point, end_point = endpoints

        # Generate evenly spaced points along the line
        t_values = np.linspace(0, 1, num_points)
        points = np.outer(1 - t_values, start_point) + np.outer(t_values, end_point)

        # Evaluate acquisition function on all points at once (expects 2D array)
        acquisition_values = penalty_acquisition(points)

        # Replace any inf values with -10000
        acquisition_values = np.where(np.isinf(acquisition_values), -10000, acquisition_values)

        # Return sum of acquisition values
        return np.sum(acquisition_values)



    def linebo_sampler(self, x_tell, zoom_bounds=None):
        """
        Simplified LineBO sampler that finds optimal direction in simplex-constrained space.

        This method implements the LineBO (Line Bayesian Optimization) algorithm for
        simplex-constrained optimization. It generates multiple candidate lines from
        the current point, evaluates their integrated acquisition values, and selects
        the most promising line for experimental evaluation.

        Args:
            x_tell (numpy.ndarray): 1D array of shape (dimensions,) containing the
                current point in the simplex. This point should satisfy the simplex
                constraint (sum = 1) and will be used as the starting point for
                line generation.
            zoom_bounds (list of tuples, optional): Current zoom bounds from the
                activation. If None, uses global bounds. Format: [(min1, max1),
                (min2, max2), ...]. These bounds define the current search space
                for line generation.

        Returns:
            tuple: (x_actual_array, y_actual_array) where:
                - x_actual_array (numpy.ndarray): 2D array of shape (n_experiments, dimensions)
                  containing the actual experimental points sampled along the selected line
                - y_actual_array (numpy.ndarray): 1D array of shape (n_experiments,) containing
                  the corresponding objective function values

        Algorithm:
            1. Validates that x_tell is on the simplex (sum = 1)
            2. Generates linebo_num_lines zero-sum direction vectors
            3. For each direction, finds line endpoints within bounds
            4. Integrates acquisition function along each valid line segment
            5. Orders lines by integrated acquisition value
            6. Calls objective_function_wrapper with ordered line endpoints
            7. Returns the experimental results from the line evaluation
        """
        # Convert inputs to numpy arrays
        x_tell = np.asarray(x_tell)

        # Use zoom bounds if provided, otherwise use global bounds
        bounds_to_use = zoom_bounds if zoom_bounds is not None else self.bounds

        # Convert bounds to numpy array format expected by _linebo_find_line_endpoints
        bounds_array = np.array(bounds_to_use)

        # Check if point is actually on simplex
        point_sum = np.sum(x_tell)
        if abs(point_sum - 1.0) > 1e-10:
            print(f"⚠️ Point not exactly on simplex! Sum: {point_sum:.10f}, adjusting...")
            x_tell = x_tell / point_sum

        print(f"Starting LineBO sampling from point: {x_tell}")

        # Step 1: Generate zero-sum directions
        directions = self._linebo_generate_equally_spaced_directions(self.linebo_num_lines)

        # Step 2: Find line endpoints for each direction
        line_endpoints = []
        for direction in directions:
            endpoints = self._linebo_find_line_endpoints(x_tell, direction, bounds_to_use)
            if endpoints is not None:
                line_endpoints.append(endpoints)

        if not line_endpoints:
            print("⚠️ No valid line segments found within bounds! Using fallback...")
            # Use center point as both endpoints for fallback
            endpoints = np.array([x_tell, x_tell])
            return self.objective_function_wrapper(endpoints)

        print(f"Found {len(line_endpoints)} valid line segments")

        # Step 3: Integrate acquisition along each line and store with endpoints
        acquisition_values = []
        for endpoints in line_endpoints:
            integrated_value = self._linebo_integrate_acquisition(
                endpoints, self.linebo_pts_per_line, self._penalty_acquisition_function
            )
            acquisition_values.append((integrated_value, endpoints))

        # Sort by integrated acquisition value (descending)
        acquisition_values.sort(key=lambda x: x[0], reverse=True)

        # Extract ordered endpoints
        ordered_endpoints = np.array([x[1] for x in acquisition_values])

        print(f"Best integrated acquisition value: {acquisition_values[0][0]:.4f}")
        print(f"Best line endpoints: {ordered_endpoints[0][0]} and {ordered_endpoints[0][1]}")

        # Step 4: Call objective_function_wrapper with ordered endpoints
        val = self.objective_function_wrapper(ordered_endpoints)
        print("received val:", val)

        return val

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
        Check convergence using the best num_points_converge points from the batch.

        This method implements robust convergence detection by evaluating multiple
        experimental points simultaneously. It identifies the best performing points
        from a batch of experimental evaluations and checks if they all have low
        prediction errors, indicating convergence to a local minimum.

        Args:
            x_actual_array (numpy.ndarray): Array of actual experimental points from
                the batch evaluation. Shape: (n_experiments, dimensions). These are
                the experimental points that were evaluated during the current iteration.
            y_actual_array (numpy.ndarray): Array of actual experimental values corresponding
                to x_actual_array. Shape: (n_experiments,). Used to identify the best
                performing points.

        Returns:
            tuple: (converged, best_x_actual, best_y_actual) where:
                - converged (bool): True if convergence is detected, False otherwise.
                  Convergence is declared when ALL best points have prediction errors
                  below the tolerance threshold.
                - best_x_actual (numpy.ndarray): Array of shape (n_best, dimensions)
                  containing the best num_points_converge experimental points
                - best_y_actual (numpy.ndarray): Array of shape (n_best,) containing
                  the corresponding objective function values

        Algorithm:
            1. Identifies the best num_points_converge points (lowest y values)
            2. Makes GP predictions for these best points
            3. Computes relative prediction errors for each point:
               - If y_actual == 0: error = abs(y_predicted)
               - Otherwise: error = abs(y_actual - y_predicted) / abs(y_actual)
            4. Checks if ALL best points have errors below tolerance
            5. Stores mean error for tracking purposes
            6. Returns convergence status and best points

        Convergence Criterion:
            - Requires ALL best points to have prediction errors below tolerance
            - Uses relative error to handle different scales of objective values
            - Provides robust convergence detection against noise
            - More stringent than single-point convergence checking

        Class Parameters Used:
            - self.num_points_converge: Number of best points to check for convergence
            - self.tolerance: Convergence tolerance threshold
            - self.gp: Trained Gaussian Process model for predictions
            - self.convergence_errors: List to track mean prediction errors

        Robustness Features:
            - Uses multiple points for convergence detection
            - Handles edge cases (zero actual values) gracefully
            - Provides detailed progress reporting
            - Stores converged points for analysis
            - More reliable than single-point convergence checking

        Notes:
            - Preferred over _check_convergence for robust convergence detection
            - Uses batch evaluation for more reliable convergence assessment
            - Provides detailed logging about convergence progress
            - Critical for accurate needle detection in the optimization process
        """
        if len(x_actual_array) == 0:
            return False, np.array([]), np.array([])

        # Get the best num_points_converge points (lowest y values)
        n_points_to_check = min(self.num_points_converge, len(y_actual_array))
        best_indices = np.argsort(y_actual_array)[:n_points_to_check]

        best_x_actual = x_actual_array[best_indices]
        best_y_actual = y_actual_array[best_indices]

        # Get GP predictions for the best points
        y_pred_batch, y_std_batch = self.gp.predict(best_x_actual, return_std=True)

        # Calculate prediction errors for the best points
        prediction_errors = []
        for y_actual, y_pred in zip(best_y_actual, y_pred_batch):
            if y_actual == 0:
                error = abs(y_pred)
            else:
                error = abs(y_actual - y_pred) / abs(y_actual)
            prediction_errors.append(error)

        prediction_errors = np.array(prediction_errors)

        # Store the mean error for tracking (backward compatibility)
        mean_prediction_error = np.mean(prediction_errors)
        self.convergence_errors.append(mean_prediction_error)

        # Simple convergence criterion: ALL best points must be below tolerance
        all_best_points_converged = np.all(prediction_errors <= self.tolerance)

        if all_best_points_converged:
            print(f"🎯 All {len(prediction_errors)} best points converged! Errors: {prediction_errors}")
            print(f"   Max error: {np.max(prediction_errors):.6f} <= tolerance: {self.tolerance}")
            return True, best_x_actual, best_y_actual

        # Show progress
        n_converged = np.sum(prediction_errors <= self.tolerance)
        print(f"📊 Batch convergence: {n_converged}/{len(prediction_errors)} best points below tolerance")
        print(f"   Mean error: {mean_prediction_error:.6f}, Max error: {np.max(prediction_errors):.6f}")

        return False, best_x_actual, best_y_actual

    def plot_convergence(self, save_path=None):
        """
        Plot convergence showing Y values collected over time with red lines marking needles found.

        Args:
            save_path (str, optional): Path to save the plot. If None, plot is displayed.
        """
        import matplotlib.pyplot as plt

        if len(self.Y_all) == 0:
            print("No data to plot - no evaluations have been performed yet.")
            return

        # Create the convergence plot
        plt.figure(figsize=(12, 6))

        # Plot all Y values over time
        x_points = np.arange(len(self.Y_all))
        plt.plot(x_points, self.Y_all, 'b-', alpha=0.7, linewidth=1, label='All Evaluations')
        plt.scatter(x_points, self.Y_all, c='blue', s=20, alpha=0.6, zorder=3)

        # Plot running minimum
        running_min = np.minimum.accumulate(self.Y_all)
        plt.plot(x_points, running_min, 'g-', linewidth=2, alpha=0.8, label='Running Minimum')

        # Add red horizontal lines for each needle found
        if len(self.needle_values) > 0:
            for i, needle_value in enumerate(self.needle_values):
                plt.axhline(y=needle_value, color='red', linestyle='-', linewidth=2, alpha=0.8,
                           label=f'Needle {i+1}' if i == 0 else "")
                # Add annotation for needle value
                plt.text(len(self.Y_all) * 0.02, needle_value, f'Needle {i+1}: {needle_value:.4f}',
                        fontsize=10, verticalalignment='bottom', color='red', weight='bold')

        # Add vertical lines to show when needles were discovered
        if len(self.needle_discovery_indices) > 0:
            for i, discovery_idx in enumerate(self.needle_discovery_indices):
                plt.axvline(x=discovery_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
                plt.text(discovery_idx, plt.ylim()[1] * 0.95, f'Needle {i+1}\nDiscovered',
                        fontsize=9, rotation=90, verticalalignment='top', color='red', weight='bold')

        # Customize the plot
        plt.xlabel('Number of Evaluations', fontsize=12)
        plt.ylabel('Objective Function Value', fontsize=12)
        plt.title('ZoMBI-Hop Convergence Plot', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Add statistics text box
        stats_text = f'Total Evaluations: {len(self.Y_all)}\n'
        stats_text += f'Needles Found: {len(self.needles)}\n'
        if len(self.needle_discovery_indices) > 0:
            stats_text += f'Discovery Points: {self.needle_discovery_indices}\n'
        if len(self.Y_all) > 0:
            stats_text += f'Best Value: {np.min(self.Y_all):.6f}\n'
            stats_text += f'Worst Value: {np.max(self.Y_all):.6f}\n'
            stats_text += f'Mean Value: {np.mean(self.Y_all):.6f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

# --- Load bandgap.csv ---
df = pd.read_csv("bandgap.csv")
X_data = df[["Cs", "MA", "FA"]].values
y_data = df["Bandgap"].values

# --- Fit Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_data, y_data)

OPTIMIZING_DIMS = [0, 1, 2]  # Using all 3 dimensions since we only have Cs, MA, FA

def normalize_last_axis(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array along its last axis so that the values sum to 1.

    Parameters
    ----------
    arr : array-like of shape (..., 3)
        Input data. Can be 1-D (3,) or 2-D (n, 3), or higher-dimensional
        as long as the last axis is length 3.

    Returns
    -------
    normalized : np.ndarray
        Array of same shape as `arr`, with values along the last axis
        scaled to sum to 1.
    """
    a = np.asarray(arr, dtype=float)
    sums = a.sum(axis=-1, keepdims=True)
    return a / sums

def objective_function(ordered_endpoints, num_experiments=24):
    """
    Evaluates points along lines using the random forest model.
    Args:
        ordered_endpoints (numpy.ndarray): Array of line endpoints ordered by acquisition value
            (greatest to smallest). Shape: (num_lines, 2, dimensions)

    Returns:
        tuple: (x_actual_array, y_actual_array) containing the predictions
    """
    # Get top 2 lines' endpoints
    best_start = ordered_endpoints[0][0]
    best_end = ordered_endpoints[0][1]
    cache_start = ordered_endpoints[1][0]
    cache_end = ordered_endpoints[1][1]
    print(ordered_endpoints)

    # Generate evenly spaced points along both lines
    x = np.array([best_start + t * (best_end - best_start) for t in np.linspace(0, 1, num_experiments)])
    x_cache = np.array([cache_start + t * (cache_end - cache_start) for t in np.linspace(0, 1, num_experiments)])

    # Normalize and round compositions
    x_norm = normalize_last_axis(np.round(x, 3))
    x_cache_norm = normalize_last_axis(np.round(x_cache, 3))

    # Combine points from both lines
    x_all = np.vstack([x_norm, x_cache_norm])

    # Get predictions from random forest
    y_pred = rf.predict(x_all)  # Negative since we're minimizing

    return x_all, y_pred

def run_zombi_main():
    np.random.seed(28)

    # instantiate zombihop class
    dimensions = 3
    num_activations = 3
    n_experiments = 24
    max_iterations = 10
    num_samples = 15000
    max_gp_points = 150

    optimizer = ZoMBIHop(
        objective_function=objective_function,
        dimensions=dimensions,
        X_init_actual=np.empty((0, dimensions)),  # Empty n=0, d=dimensions array
        Y_init=np.empty((0,)),  # Empty n=0 array
        num_activations=num_activations,
        max_gp_points=max_gp_points,
        penalty_zoom_percentage=0.6,
        max_iterations=max_iterations,
        tolerance=1e-2,
        top_m_points=max(4, dimensions),
        num_samples=num_samples,
        max_zoom_levels=2,
        num_experiments=n_experiments,
        linebo_num_lines=max(30, 10 * dimensions),
        linebo_pts_per_line=max(50, 20 * dimensions),
        num_points_converge=3,
        radius_multiplier=10,
    )

    num_start_points = 1

    start_points = optimizer.sample_bounded_simplex_cfs(np.zeros(dimensions), np.ones(dimensions), 1.0, num_start_points)

    print(start_points)

    X_init_actual = np.zeros((0, dimensions))  # Empty n=0, d=dimensions array
    Y_init = np.zeros((0,))  # Empty n=0 array

    for point in start_points:
        # Ensure point is a CPU numpy array
        if hasattr(point, 'cpu'):
            point = point.cpu().numpy()
        X_actual, Y_actual = optimizer.linebo_sampler(point)
        print("got: ", X_actual, Y_actual)
        X_init_actual = np.vstack((X_init_actual, X_actual))
        Y_init = np.append(Y_init, Y_actual)

    print(X_init_actual, Y_init)
    print("--------------------------------")
    print("should be the same as above:")
    print(optimizer.X_all_actual, optimizer.Y_all)

    # Run the main ZoMBI-Hop optimization
    results = optimizer.run_zombi_hop()

    # Plot convergence at the end
    print("\n" + "="*60)
    print("📊 GENERATING CONVERGENCE PLOT")
    print("="*60)

    # Ensure figs directory exists
    import os
    os.makedirs("./figs", exist_ok=True)

    optimizer.plot_convergence(save_path="./figs/zombi_convergence_plot.png")

    print(results)


if __name__ == "__main__":
    run_zombi_main()
