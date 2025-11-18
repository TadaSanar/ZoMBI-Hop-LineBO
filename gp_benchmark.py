import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
import json
import os
from typing import List, Tuple, Dict

# CUDA optimization settings (same as zombihop_linebo_new.py)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class GPBenchmark:
    """
    Benchmark GP performance to determine optimal max_gp_points for ZoMBIHop.
    Replicates the exact GP setup from zombihop_linebo_new.py.
    """
    
    def __init__(self, device='cuda', dtype=torch.float64, dimensions=10):
        self.device = torch.device(device)
        self.dtype = dtype
        self.dimensions = dimensions
        
        # CUDA optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"Initialized GP Benchmark on CUDA device: {torch.cuda.get_device_name()}")
            print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def generate_synthetic_data(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic training data on simplex (same distribution as ZoMBIHop).
        Uses the random_simplex method from zombihop_linebo_new.py.
        """
        # Create bounds tensor
        bounds = torch.zeros((2, self.dimensions), device=self.device, dtype=self.dtype)
        bounds[0] = 0.0  # Lower bounds
        bounds[1] = 1.0  # Upper bounds
        
        # Generate random simplex points
        X = self.random_simplex(n_points, bounds[0], bounds[1])
        
        # Generate synthetic Y values with realistic noise
        # Use a multi-modal function similar to the test functions in zombihop_linebo_new.py
        Y = self.synthetic_objective(X)
        
        return X, Y
    
    def random_simplex(self, num_samples: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Simplified version of random_simplex from zombihop_linebo_new.py for benchmarking.
        """
        # For benchmarking, use a simpler Dirichlet-based approach
        alpha = torch.ones(self.dimensions, device=self.device, dtype=self.dtype)
        samples = torch.distributions.Dirichlet(alpha).sample((num_samples,))
        
        # Scale to bounds
        samples = a + samples * (b - a)
        
        # Ensure they sum to 1 (simplex constraint)
        samples = samples / samples.sum(dim=1, keepdim=True)
        
        return samples.to(device=self.device, dtype=self.dtype)
    
    def synthetic_objective(self, X: torch.Tensor) -> torch.Tensor:
        """
        Create a synthetic multi-modal objective function for benchmarking.
        """
        # Multi-modal function with noise (similar to the test functions)
        Y = torch.zeros(X.shape[0], 1, device=self.device, dtype=self.dtype)
        
        # Add multiple modes
        centers = torch.tensor([
            [0.4, 0.0, 0.1, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5],
            [0.0, 0.1, 0.2, 0.1, 0.5, 0.1, 0.1, 0.0, 0.1, 0.0]
        ], device=self.device, dtype=self.dtype)
        
        for i, center in enumerate(centers):
            distances = torch.norm(X - center.unsqueeze(0), dim=1)
            Y[:, 0] += (2.0 + i) * torch.exp(-10.0 * distances**2)
        
        # Add noise
        noise = torch.randn_like(Y) * 0.1
        Y = Y + noise
        
        return Y
    
    def create_projected_acquisition(self, gp, best_f: float) -> nn.Module:
        """
        Create the same ProjectedAcq wrapper used in zombihop_linebo_new.py.
        """
        acq_fn = LogExpectedImprovement(gp, best_f=best_f)
        
        class ProjectedAcq(torch.nn.Module):
            def __init__(self, base, proj_fn, penalty_value: float = 0.0):
                super().__init__()
                self.base = base
                self.proj_fn = proj_fn
                self.penalty_value = penalty_value
            
            def forward(self, Xq: torch.Tensor) -> torch.Tensor:
                # Simple projection to simplex (simplified version)
                X_proj = self.proj_simplex(Xq)
                return self.base(X_proj)
            
            def proj_simplex(self, X):
                """Simplified simplex projection for benchmarking."""
                if X.dim() == 3:
                    n, l, d = X.shape
                    X_2d = X.reshape(-1, d)
                else:
                    X_2d = X
                
                # Clamp to positive values and normalize
                X_proj = torch.clamp(X_2d, min=0.0)
                X_proj = X_proj / X_proj.sum(dim=-1, keepdim=True)
                
                if X.dim() == 3:
                    X_proj = X_proj.reshape(n, l, d)
                
                return X_proj
        
        return ProjectedAcq(acq_fn, None)
    
    def benchmark_gp_fitting(self, training_sizes: List[int], num_repeats: int = 3) -> Dict:
        """
        Benchmark GP fitting time for different training set sizes.
        """
        results = {
            'training_sizes': training_sizes,
            'fit_times': [],
            'fit_times_std': [],
            'memory_usage': []
        }
        
        print("Benchmarking GP fitting times...")
        
        for n_train in training_sizes:
            fit_times = []
            
            for repeat in range(num_repeats):
                # Generate training data
                X_train, Y_train = self.generate_synthetic_data(n_train)
                
                # Clear GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Time GP fitting
                start_time = time.time()
                
                gp = SingleTaskGP(X_train, Y_train)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                
                fit_time = time.time() - start_time
                fit_times.append(fit_time)
                
                # Clean up
                del gp, mll
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_fit_time = np.mean(fit_times)
            std_fit_time = np.std(fit_times)
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_gb = torch.cuda.memory_allocated() / 1024**3
            else:
                memory_gb = 0.0
            
            results['fit_times'].append(avg_fit_time)
            results['fit_times_std'].append(std_fit_time)
            results['memory_usage'].append(memory_gb)
            
            print(f"Training size {n_train}: {avg_fit_time:.3f} ± {std_fit_time:.3f}s, Memory: {memory_gb:.2f} GB")
        
        return results
    
    def benchmark_gp_inference(self, training_sizes: List[int], n_candidates: int = 10000, 
                             num_repeats: int = 3) -> Dict:
        """
        Benchmark GP inference time on n_candidates points for different training set sizes.
        This is the key benchmark for determining the max_gp_points upper bound.
        """
        results = {
            'training_sizes': training_sizes,
            'inference_times': [],
            'inference_times_std': [],
            'acquisition_times': [],
            'acquisition_times_std': [],
            'memory_peak': []
        }
        
        print(f"Benchmarking GP inference on {n_candidates} candidates...")
        
        # Generate candidate points once
        bounds = torch.zeros((2, self.dimensions), device=self.device, dtype=self.dtype)
        bounds[0] = 0.0
        bounds[1] = 1.0
        X_candidates = self.random_simplex(n_candidates, bounds[0], bounds[1])
        X_candidates_3d = X_candidates.unsqueeze(1)  # Shape for acquisition function
        
        for n_train in training_sizes:
            inference_times = []
            acquisition_times = []
            memory_peaks = []
            
            for repeat in range(num_repeats):
                # Generate training data
                X_train, Y_train = self.generate_synthetic_data(n_train)
                
                # Fit GP
                gp = SingleTaskGP(X_train, Y_train)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                
                # Create acquisition function
                best_f = Y_train.max().item()
                acq = self.create_projected_acquisition(gp, best_f)
                
                # Clear GPU memory and measure peak
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Time inference
                start_time = time.time()
                
                with torch.no_grad():
                    # This is the same pattern as in zombihop_linebo_new.py line 382
                    acq_values = acq(X_candidates_3d).squeeze()
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Measure peak memory
                if self.device.type == 'cuda':
                    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                    memory_peaks.append(peak_memory_gb)
                else:
                    memory_peaks.append(0.0)
                
                # Time acquisition function evaluation separately (more detailed timing)
                start_time = time.time()
                
                with torch.no_grad():
                    # Simulate the full acquisition evaluation process
                    for i in range(0, n_candidates, 1000):  # Process in batches like the real code
                        end_idx = min(i + 1000, n_candidates)
                        batch = X_candidates_3d[i:end_idx]
                        _ = acq(batch)
                
                acquisition_time = time.time() - start_time
                acquisition_times.append(acquisition_time)
                
                # Clean up
                del gp, mll, acq
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            avg_acquisition_time = np.mean(acquisition_times)
            std_acquisition_time = np.std(acquisition_times)
            avg_memory_peak = np.mean(memory_peaks)
            
            results['inference_times'].append(avg_inference_time)
            results['inference_times_std'].append(std_inference_time)
            results['acquisition_times'].append(avg_acquisition_time)
            results['acquisition_times_std'].append(std_acquisition_time)
            results['memory_peak'].append(avg_memory_peak)
            
            print(f"Training size {n_train}:")
            print(f"  Inference: {avg_inference_time:.3f} ± {std_inference_time:.3f}s")
            print(f"  Acquisition: {avg_acquisition_time:.3f} ± {std_acquisition_time:.3f}s")
            print(f"  Peak Memory: {avg_memory_peak:.2f} GB")
            
            # Check if we exceed 60s constraint
            if avg_inference_time > 60.0:
                print(f"  ⚠️  EXCEEDS 60s CONSTRAINT!")
        
        return results
    
    def find_optimal_max_gp_points(self, time_limit: float = 60.0) -> int:
        """
        Binary search to find the optimal max_gp_points that stays under time_limit.
        """
        print(f"\nFinding optimal max_gp_points for {time_limit}s time limit...")
        
        # Generate candidate points
        bounds = torch.zeros((2, self.dimensions), device=self.device, dtype=self.dtype)
        bounds[0] = 0.0
        bounds[1] = 1.0
        X_candidates = self.random_simplex(10000, bounds[0], bounds[1])
        X_candidates_3d = X_candidates.unsqueeze(1)
        
        # Binary search bounds
        low, high = 50, 5000
        optimal_size = low
        
        while low <= high:
            mid = (low + high) // 2
            
            # Test this size
            X_train, Y_train = self.generate_synthetic_data(mid)
            
            # Fit GP
            gp = SingleTaskGP(X_train, Y_train)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            # Create acquisition function
            best_f = Y_train.max().item()
            acq = self.create_projected_acquisition(gp, best_f)
            
            # Time inference
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start_time = time.time()
            
            with torch.no_grad():
                acq_values = acq(X_candidates_3d).squeeze()
            
            inference_time = time.time() - start_time
            
            print(f"Testing {mid} training points: {inference_time:.3f}s")
            
            if inference_time <= time_limit:
                optimal_size = mid
                low = mid + 1
            else:
                high = mid - 1
            
            # Clean up
            del gp, mll, acq
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\nOptimal max_gp_points: {optimal_size}")
        return optimal_size
    
    def run_full_benchmark(self, save_results: bool = True) -> Dict:
        """
        Run the complete benchmark suite.
        """
        print("="*60)
        print("GP BENCHMARK FOR ZOMBIHOP")
        print("="*60)
        
        # Define training sizes to test
        training_sizes = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
        
        # Run benchmarks
        fitting_results = self.benchmark_gp_fitting(training_sizes, num_repeats=3)
        inference_results = self.benchmark_gp_inference(training_sizes, n_candidates=10000, num_repeats=3)
        
        # Find optimal size
        optimal_size = self.find_optimal_max_gp_points(time_limit=60.0)
        
        # Combine results
        full_results = {
            'benchmark_config': {
                'device': str(self.device),
                'dtype': str(self.dtype),
                'dimensions': self.dimensions,
                'n_candidates': 10000,
                'time_limit': 60.0
            },
            'fitting': fitting_results,
            'inference': inference_results,
            'optimal_max_gp_points': optimal_size,
            'gpu_info': {
                'name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            }
        }
        
        if save_results:
            # Save results
            os.makedirs('benchmark_results', exist_ok=True)
            with open('benchmark_results/gp_benchmark_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_for_json(full_results)
                json.dump(json_results, f, indent=2)
            
            # Create plots
            self.create_plots(full_results)
        
        return full_results
    
    def _prepare_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def create_plots(self, results: Dict):
        """Create visualization plots for the benchmark results."""
        print("Creating benchmark plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        training_sizes = results['fitting']['training_sizes']
        
        # Plot 1: GP Fitting Time vs Training Size
        ax1.errorbar(training_sizes, results['fitting']['fit_times'], 
                     yerr=results['fitting']['fit_times_std'], 
                     marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('GP Fitting Time (s)')
        ax1.set_title('GP Fitting Time vs Training Set Size')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Inference Time vs Training Size
        ax2.errorbar(training_sizes, results['inference']['inference_times'],
                     yerr=results['inference']['inference_times_std'],
                     marker='s', color='red', capsize=5, capthick=2)
        ax2.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='60s Limit')
        ax2.axvline(x=results['optimal_max_gp_points'], color='green', linestyle='--', 
                    alpha=0.7, label=f'Optimal Size: {results["optimal_max_gp_points"]}')
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('Inference Time on 10k Candidates (s)')
        ax2.set_title('GP Inference Time vs Training Set Size')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Plot 3: Memory Usage
        ax3.plot(training_sizes, results['inference']['memory_peak'], 
                 marker='^', color='purple')
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('Peak GPU Memory (GB)')
        ax3.set_title('GPU Memory Usage vs Training Set Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Acquisition Function Time
        ax4.errorbar(training_sizes, results['inference']['acquisition_times'],
                     yerr=results['inference']['acquisition_times_std'],
                     marker='d', color='orange', capsize=5, capthick=2)
        ax4.set_xlabel('Training Set Size')
        ax4.set_ylabel('Acquisition Function Time (s)')
        ax4.set_title('Acquisition Function Evaluation Time')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('benchmark_results/gp_benchmark_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plots saved to benchmark_results/gp_benchmark_plots.png")

def main():
    """Run the GP benchmark."""
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU (will be much slower)")
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Create and run benchmark
    benchmark = GPBenchmark(device=device, dtype=torch.float64, dimensions=10)
    results = benchmark.run_full_benchmark(save_results=True)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Optimal max_gp_points for 60s constraint: {results['optimal_max_gp_points']}")
    print(f"GPU: {results['gpu_info']['name']}")
    print(f"Total GPU Memory: {results['gpu_info']['memory_gb']:.1f} GB")
    
    # Find the inference time for the optimal size
    training_sizes = results['inference']['training_sizes']
    inference_times = results['inference']['inference_times']
    
    # Find closest training size to optimal
    optimal_idx = min(range(len(training_sizes)), 
                     key=lambda i: abs(training_sizes[i] - results['optimal_max_gp_points']))
    optimal_time = inference_times[optimal_idx]
    
    print(f"Inference time at optimal size: {optimal_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main() 