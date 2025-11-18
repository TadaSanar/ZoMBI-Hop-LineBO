"""
Script to analyze ZoMBI-Hop results and find optimal parameters.
Finds parameters that minimize average distance with high consistency.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def load_results(results_dir: str = "results") -> List[Dict]:
    """Load all result JSON files from the specified directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Directory {results_dir} does not exist!")
        return results
    
    json_files = list(results_path.glob("*.json"))
    print(f"Loading {len(json_files)} result files from {results_dir}...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['filename'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Successfully loaded {len(results)} result files.")
    return results


def calculate_metrics(results: List[Dict]) -> pd.DataFrame:
    """Calculate performance metrics for each result."""
    data = []
    
    for result in results:
        params = result.get('params', {})
        distances = result.get('all_distances', [])
        
        if not distances:
            continue
        
        # Calculate metrics
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        median_distance = np.median(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        cv = std_distance / avg_distance if avg_distance > 0 else float('inf')  # Coefficient of variation
        q25 = np.percentile(distances, 25)
        q75 = np.percentile(distances, 75)
        iqr = q75 - q25
        
        # Combined score: lower is better
        # Weight both low average and low variance
        combined_score = avg_distance + 0.5 * std_distance
        
        # Another scoring option: penalize inconsistency more
        consistency_score = avg_distance * (1 + cv)
        
        row = {
            'filename': result.get('filename', 'unknown'),
            'avg_distance': avg_distance,
            'std_distance': std_distance,
            'median_distance': median_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'cv': cv,  # Lower CV = more consistent
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'combined_score': combined_score,
            'consistency_score': consistency_score,
            'n_samples': len(distances),
        }
        
        # Add all parameters
        for key, value in params.items():
            row[f'param_{key}'] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def print_top_results(df: pd.DataFrame, n: int = 10):
    """Print top N results by different criteria."""
    print("\n" + "="*80)
    print(f"TOP {n} RESULTS BY DIFFERENT CRITERIA")
    print("="*80)
    
    # Top by average distance
    print(f"\n{'='*80}")
    print(f"TOP {n} BY AVERAGE DISTANCE (Lower is Better)")
    print("="*80)
    top_avg = df.nsmallest(n, 'avg_distance')
    print(top_avg[['filename', 'avg_distance', 'std_distance', 'cv', 'min_distance', 'max_distance']].to_string(index=False))
    
    # Top by consistency (lowest CV)
    print(f"\n{'='*80}")
    print(f"TOP {n} BY CONSISTENCY (Lowest Coefficient of Variation)")
    print("="*80)
    top_cv = df.nsmallest(n, 'cv')
    print(top_cv[['filename', 'avg_distance', 'std_distance', 'cv', 'min_distance', 'max_distance']].to_string(index=False))
    
    # Top by combined score (balance of both)
    print(f"\n{'='*80}")
    print(f"TOP {n} BY COMBINED SCORE (Balances Average & Consistency)")
    print("="*80)
    top_combined = df.nsmallest(n, 'combined_score')
    print(top_combined[['filename', 'avg_distance', 'std_distance', 'cv', 'combined_score']].to_string(index=False))
    
    # Top by consistency score
    print(f"\n{'='*80}")
    print(f"TOP {n} BY CONSISTENCY SCORE (Penalizes Variance More)")
    print("="*80)
    top_consistency = df.nsmallest(n, 'consistency_score')
    print(top_consistency[['filename', 'avg_distance', 'std_distance', 'cv', 'consistency_score']].to_string(index=False))


def print_best_params(df: pd.DataFrame, criterion: str = 'combined_score'):
    """Print the best parameter configuration."""
    best_idx = df[criterion].idxmin()
    best_result = df.loc[best_idx]
    
    print(f"\n{'='*80}")
    print(f"BEST PARAMETER CONFIGURATION (by {criterion})")
    print("="*80)
    print(f"\nFilename: {best_result['filename']}")
    print(f"Average Distance: {best_result['avg_distance']:.6f}")
    print(f"Std Distance: {best_result['std_distance']:.6f}")
    print(f"Coefficient of Variation: {best_result['cv']:.4f}")
    print(f"Min Distance: {best_result['min_distance']:.6f}")
    print(f"Max Distance: {best_result['max_distance']:.6f}")
    print(f"Combined Score: {best_result['combined_score']:.6f}")
    print(f"Consistency Score: {best_result['consistency_score']:.6f}")
    
    print("\nParameters:")
    param_cols = [col for col in df.columns if col.startswith('param_')]
    for col in sorted(param_cols):
        param_name = col.replace('param_', '')
        print(f"  {param_name}: {best_result[col]}")
    
    return best_result


def analyze_parameter_correlations(df: pd.DataFrame):
    """Analyze which parameters correlate with better performance."""
    print(f"\n{'='*80}")
    print("PARAMETER CORRELATIONS WITH PERFORMANCE")
    print("="*80)
    
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    if not param_cols:
        print("No parameter columns found!")
        return
    
    # Correlations with average distance
    print("\nCorrelation with Average Distance (negative = parameter helps reduce distance):")
    corr_avg = df[param_cols + ['avg_distance']].corr()['avg_distance'].drop('avg_distance').sort_values()
    print(corr_avg.to_string())
    
    # Correlations with CV (consistency)
    print("\nCorrelation with Coefficient of Variation (negative = parameter helps consistency):")
    corr_cv = df[param_cols + ['cv']].corr()['cv'].drop('cv').sort_values()
    print(corr_cv.to_string())
    
    # Correlations with combined score
    print("\nCorrelation with Combined Score (negative = parameter helps overall):")
    corr_combined = df[param_cols + ['combined_score']].corr()['combined_score'].drop('combined_score').sort_values()
    print(corr_combined.to_string())


def create_visualizations(df: pd.DataFrame, output_dir: str = 'analysis_plots'):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. Scatter plot: Average distance vs Std distance
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['avg_distance'], df['std_distance'], 
                         c=df['combined_score'], cmap='viridis_r', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Combined Score (lower is better)')
    plt.xlabel('Average Distance')
    plt.ylabel('Standard Deviation')
    plt.title('Average Distance vs Consistency')
    
    # Mark the best point
    best_idx = df['combined_score'].idxmin()
    plt.scatter(df.loc[best_idx, 'avg_distance'], 
               df.loc[best_idx, 'std_distance'],
               color='red', s=200, marker='*', 
               edgecolors='black', linewidth=2,
               label='Best (Combined Score)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distance_vs_consistency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/distance_vs_consistency.png")
    plt.close()
    
    # 2. Distribution of performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['avg_distance'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['avg_distance'].median(), color='red', linestyle='--', label='Median')
    axes[0, 0].set_xlabel('Average Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Average Distance')
    axes[0, 0].legend()
    
    axes[0, 1].hist(df['std_distance'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(df['std_distance'].median(), color='red', linestyle='--', label='Median')
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Std Deviation')
    axes[0, 1].legend()
    
    axes[1, 0].hist(df['cv'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].axvline(df['cv'].median(), color='red', linestyle='--', label='Median')
    axes[1, 0].set_xlabel('Coefficient of Variation')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of CV (Consistency)')
    axes[1, 0].legend()
    
    axes[1, 1].hist(df['combined_score'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(df['combined_score'].median(), color='red', linestyle='--', label='Median')
    axes[1, 1].set_xlabel('Combined Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Combined Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/performance_distributions.png")
    plt.close()
    
    # 3. Parameter correlation heatmap
    param_cols = [col for col in df.columns if col.startswith('param_')]
    metric_cols = ['avg_distance', 'std_distance', 'cv', 'combined_score']
    
    if param_cols:
        corr_matrix = df[param_cols + metric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Parameter Correlations with Performance Metrics')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/correlation_heatmap.png")
        plt.close()


def save_analysis_csv(df: pd.DataFrame, output_file: str = 'parameter_analysis.csv'):
    """Save the full analysis to a CSV file."""
    df_sorted = df.sort_values('combined_score')
    df_sorted.to_csv(output_file, index=False)
    print(f"\nSaved full analysis to: {output_file}")


def main():
    """Main analysis function."""
    print("="*80)
    print("ZOMBI-HOP PARAMETER OPTIMIZATION ANALYSIS")
    print("="*80)
    
    # Load results
    results = load_results("results")
    
    if not results:
        print("No results found! Exiting.")
        return
    
    # Calculate metrics
    df = calculate_metrics(results)
    
    print(f"\nAnalyzing {len(df)} parameter configurations...")
    print(f"Average of all avg_distances: {df['avg_distance'].mean():.6f}")
    print(f"Best avg_distance found: {df['avg_distance'].min():.6f}")
    print(f"Worst avg_distance found: {df['avg_distance'].max():.6f}")
    
    # Print top results
    print_top_results(df, n=10)
    
    # Print best parameters by different criteria
    print("\n")
    best_combined = print_best_params(df, criterion='combined_score')
    
    print("\n")
    best_avg = print_best_params(df, criterion='avg_distance')
    
    print("\n")
    best_cv = print_best_params(df, criterion='cv')
    
    # Analyze parameter correlations
    analyze_parameter_correlations(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Save to CSV
    save_analysis_csv(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nRecommendation: Use the parameters from the 'Best by Combined Score' configuration")
    print("for a good balance between low average distance and consistency.")
    print("\nIf you prioritize:")
    print("  - Lowest average distance: use 'Best by Average Distance' params")
    print("  - Maximum consistency: use 'Best by CV' params")


if __name__ == "__main__":
    main()




