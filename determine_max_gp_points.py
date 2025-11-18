#!/usr/bin/env python3
"""
Utility script to determine optimal max_gp_points for ZoMBIHop based on available GPU memory.

Based on benchmark results from GP_BENCHMARK_RESULTS.md
"""

import torch
import argparse

# Benchmark-derived memory usage (GB) for different training set sizes
# Format: training_points -> peak_memory_gb
MEMORY_USAGE_DATA = {
    50: 0.19,
    100: 0.36,
    200: 0.70,
    300: 1.04,
    500: 1.71,
    750: 2.57,
    1000: 3.43,
    1500: 5.11,
    2000: 6.85,
    3000: 10.26,
    4000: 14.15,
    5000: 18.66
}

# Inference times (seconds) for 10k candidates
INFERENCE_TIMES = {
    50: 0.925,
    100: 0.015,
    200: 0.023,
    300: 0.033,
    500: 0.055,
    750: 0.085,
    1000: 0.117,
    1500: 0.190,
    2000: 0.274,
    3000: 0.486,
    4000: 0.759,
    5000: 1.169
}

def get_gpu_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return None, 0, 0
    
    device_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    available_memory = (total_memory - torch.cuda.memory_allocated() / (1024**3))  # GB
    
    return device_name, total_memory, available_memory

def recommend_max_gp_points(available_memory_gb: float, safety_margin: float = 0.8, 
                           time_limit: float = 60.0) -> dict:
    """
    Recommend optimal max_gp_points based on available GPU memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        safety_margin: Safety margin (0.8 = use 80% of available memory)
        time_limit: Maximum acceptable inference time in seconds
        
    Returns:
        Dictionary with recommendations
    """
    usable_memory = available_memory_gb * safety_margin
    
    # Find largest training set size that fits in memory and meets time constraint
    recommended_points = 50  # minimum
    recommended_memory = 0
    recommended_time = 0
    
    for points, memory_gb in sorted(MEMORY_USAGE_DATA.items()):
        inference_time = INFERENCE_TIMES[points]
        
        if memory_gb <= usable_memory and inference_time <= time_limit:
            recommended_points = points
            recommended_memory = memory_gb
            recommended_time = inference_time
        else:
            break
    
    # Calculate performance improvement vs default (200 points)
    default_points = 200
    improvement_factor = recommended_points / default_points
    
    return {
        'recommended_max_gp_points': recommended_points,
        'expected_memory_usage_gb': recommended_memory,
        'expected_inference_time_s': recommended_time,
        'improvement_factor': improvement_factor,
        'usable_memory_gb': usable_memory,
        'safety_margin': safety_margin
    }

def print_recommendations(gpu_name: str, total_memory: float, available_memory: float,
                         conservative_rec: dict, aggressive_rec: dict):
    """Print formatted recommendations."""
    print("="*70)
    print("🚀 ZoMBIHop GP Performance Recommendations")
    print("="*70)
    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {total_memory:.1f} GB")
    print(f"Available Memory: {available_memory:.1f} GB")
    print()
    
    print("📊 CONSERVATIVE RECOMMENDATION (80% memory usage):")
    print(f"  max_gp_points: {conservative_rec['recommended_max_gp_points']}")
    print(f"  Expected memory usage: {conservative_rec['expected_memory_usage_gb']:.2f} GB")
    print(f"  Expected inference time: {conservative_rec['expected_inference_time_s']:.3f}s")
    print(f"  Improvement over default (200): {conservative_rec['improvement_factor']:.1f}x")
    print()
    
    print("🚀 AGGRESSIVE RECOMMENDATION (95% memory usage):")
    print(f"  max_gp_points: {aggressive_rec['recommended_max_gp_points']}")
    print(f"  Expected memory usage: {aggressive_rec['expected_memory_usage_gb']:.2f} GB")
    print(f"  Expected inference time: {aggressive_rec['expected_inference_time_s']:.3f}s")
    print(f"  Improvement over default (200): {aggressive_rec['improvement_factor']:.1f}x")
    print()
    
    print("💡 USAGE RECOMMENDATIONS:")
    print("  - Start with conservative setting for production")
    print("  - Use aggressive setting if you have dedicated GPU")
    print("  - Monitor GPU memory usage during optimization")
    print("  - Consider dynamic scaling based on optimization phase")
    print()
    
    print("⚡ PERFORMANCE NOTES:")
    print("  - All recommendations stay well under 60s inference time limit")
    print("  - Memory usage scales approximately linearly with training points")
    print("  - GP fitting time grows faster than inference time")
    print("  - Larger training sets generally improve optimization quality")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Determine optimal max_gp_points for ZoMBIHop')
    parser.add_argument('--memory', type=float, help='Override available memory (GB)')
    parser.add_argument('--time-limit', type=float, default=60.0, 
                       help='Maximum inference time limit (seconds)')
    parser.add_argument('--conservative-margin', type=float, default=0.8,
                       help='Conservative safety margin (default: 0.8)')
    parser.add_argument('--aggressive-margin', type=float, default=0.95,
                       help='Aggressive safety margin (default: 0.95)')
    
    args = parser.parse_args()
    
    # Get GPU information
    gpu_name, total_memory, available_memory = get_gpu_info()
    
    if gpu_name is None:
        print("❌ No CUDA-capable GPU found. ZoMBIHop requires GPU for optimal performance.")
        return
    
    # Override memory if specified
    if args.memory:
        available_memory = args.memory
        print(f"Using override available memory: {available_memory:.1f} GB")
    
    # Get recommendations
    conservative_rec = recommend_max_gp_points(
        available_memory, args.conservative_margin, args.time_limit
    )
    
    aggressive_rec = recommend_max_gp_points(
        available_memory, args.aggressive_margin, args.time_limit
    )
    
    # Print results
    print_recommendations(gpu_name, total_memory, available_memory, 
                         conservative_rec, aggressive_rec)
    
    # Generate code snippet
    print("📝 CODE SNIPPET:")
    print("Add this to your ZoMBIHop initialization:")
    print()
    print("```python")
    print("# Conservative setting")
    print(f"zombihop = ZoMBIHop(..., max_gp_points={conservative_rec['recommended_max_gp_points']})")
    print()
    print("# Aggressive setting")  
    print(f"zombihop = ZoMBIHop(..., max_gp_points={aggressive_rec['recommended_max_gp_points']})")
    print("```")

if __name__ == "__main__":
    main() 