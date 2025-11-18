"""
Analysis tools for ZoMBIHop trial results with variable number of minima.
"""

import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def list_trials(trial_dir=None):
    """List all available trial runs."""
    if trial_dir is None:
        # Auto-detect trial directories
        trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
        if not trial_dirs:
            print("No trial directories found")
            return []
        
        print("Found trial directories:")
        for i, td in enumerate(trial_dirs):
            print(f"  {i}: {td.name}")
        
        # Use the first one by default, or let user choose
        trial_path = trial_dirs[0]
        print(f"Using: {trial_path.name}")
    else:
        trial_path = Path(trial_dir)
    
    checkpoints_dir = trial_path / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory found at {checkpoints_dir}")
        return []
    
    runs = []
    for run_dir in checkpoints_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            uuid = run_dir.name.replace('run_', '')
            
            # Load current state
            current_state_file = run_dir / 'current_state.txt'
            if current_state_file.exists():
                with open(current_state_file, 'r') as f:
                    current_state = f.read().strip()
            else:
                current_state = "unknown"
            
            # Load config
            config_file = run_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Load latest stats
            state_dir = run_dir / 'states' / current_state
            stats = {}
            if state_dir.exists():
                stats_file = state_dir / 'stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
            
            runs.append({
                'uuid': uuid,
                'current_state': current_state,
                'num_points': stats.get('num_points_total', 0),
                'num_needles': stats.get('num_needles', 0),
                'best_value': stats.get('best_value', None),
            })
    
    print("\nAvailable trial runs:")
    print("="*80)
    for run in runs:
        print(f"UUID: {run['uuid']}")
        print(f"  State: {run['current_state']}")
        print(f"  Points: {run['num_points']}, Needles: {run['num_needles']}, Best: {run['best_value']}")
        print()
    
    return runs


def analyze_trial(uuid, trial_dir=None, plot=True):
    """
    Analyze a specific trial run.
    
    Args:
        uuid: Trial UUID
        trial_dir: Base trial directory (auto-detected if None)
        plot: Whether to generate plots
    """
    print("="*80)
    print(f"ANALYZING TRIAL: {uuid}")
    print("="*80 + "\n")
    
    if trial_dir is None:
        # Auto-detect trial directory
        trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
        if not trial_dirs:
            print("No trial directories found")
            return
        
        # Find the one containing this UUID
        trial_path = None
        for td in trial_dirs:
            checkpoints_dir = td / 'checkpoints'
            if checkpoints_dir.exists():
                run_dir = checkpoints_dir / f'run_{uuid}'
                if run_dir.exists():
                    trial_path = td
                    break
        
        if trial_path is None:
            print(f"Trial {uuid} not found in any trial directory")
            return
        
        print(f"Found trial in: {trial_path.name}")
    else:
        trial_path = Path(trial_dir)
    run_dir = trial_path / 'checkpoints' / f'run_{uuid}'
    
    if not run_dir.exists():
        print(f"Trial {uuid} not found at {run_dir}")
        return
    
    # Load config
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load metadata to get trial parameters
    metadata_file = trial_path / 'trial_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        num_minima = metadata.get('num_minima', 'unknown')
        dimensions = metadata.get('dimensions', 'unknown')
        time_limit = metadata.get('time_limit_hours', 'unknown')
        print(f"Trial parameters: {num_minima} minima, {dimensions}D, {time_limit}h limit")
    else:
        print("Warning: trial_metadata.json not found")
        num_minima = 'unknown'
    
    # Load minima locations
    minima_file = trial_path / 'minima_locations.pt'
    if minima_file.exists():
        minima_locs = torch.load(minima_file, map_location='cpu')
        print(f"True minima: {minima_locs.shape[0]}")
    else:
        print("Warning: minima_locations.pt not found")
        minima_locs = None
    
    # Collect data from all states
    states_dir = run_dir / 'states'
    state_data = []
    
    for state_dir in sorted(states_dir.iterdir()):
        if state_dir.is_dir():
            stats_file = state_dir / 'stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    stats['state_name'] = state_dir.name
                    state_data.append(stats)
    
    if not state_data:
        print("No state data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(state_data)
    
    if len(df) == 0:
        print("No state data found")
        return
    
    # Convert timestamps to hours
    if 'timestamp' in df.columns:
        start_time = df['timestamp'].min()
        df['hours'] = (df['timestamp'] - start_time) / 3600.0
    else:
        print("Warning: No timestamp data found, using sequential hours")
        df['hours'] = range(len(df))
    
    print(f"\nProgress Summary:")
    print(f"  Total states saved: {len(df)}")
    print(f"  Duration: {df['hours'].max():.2f} hours")
    print(f"  Final points: {df['num_points_total'].iloc[-1]}")
    print(f"  Final needles: {df['num_needles'].iloc[-1]}")
    print(f"  Best value: {df['best_value'].max():.6f}")
    print()
    
    # Analyze needle finding
    needle_changes = df[df['num_needles'].diff() > 0]
    if len(needle_changes) > 0:
        print(f"Needles found at:")
        for idx, row in needle_changes.iterrows():
            print(f"  {row['hours']:.2f}h - Needle {int(row['num_needles'])} (state: {row['state_name']})")
        print()
    
    # Load final state
    current_state_file = run_dir / 'current_state.txt'
    if not current_state_file.exists():
        print("Error: current_state.txt not found")
        return
    
    with open(current_state_file, 'r') as f:
        final_state_name = f.read().strip()
    
    final_state_dir = states_dir / final_state_name
    
    if not final_state_dir.exists():
        print(f"Error: Final state directory {final_state_dir} not found")
        return
    
    tensors_file = final_state_dir / 'tensors.pt'
    if not tensors_file.exists():
        print(f"Error: tensors.pt not found in {final_state_dir}")
        return
    
    print(f"Loading final state: {final_state_name}")
    tensors = torch.load(tensors_file, map_location='cpu')
    
    needles = tensors['needles']
    needle_vals = tensors['needle_vals']
    X_all_actual = tensors['X_all_actual']
    Y_all = tensors['Y_all']
    distances_input = tensors['distances']
    
    print(f"  Total points sampled: {X_all_actual.shape[0]}")
    print(f"  Needles found: {needles.shape[0]}")
    print(f"  Best objective value: {Y_all.max().item():.6f}")
    print(f"  Mean input noise (expected vs actual): {distances_input.mean().item():.6f}")
    print(f"  Median input noise: {distances_input.median().item():.6f}")
    print()
    
    # Evaluate needles against true minima
    if minima_locs is not None and needles.shape[0] > 0:
        print("Evaluating needles against true minima:")
        
        # Calculate pairwise distances
        distances_to_minima = torch.cdist(needles, minima_locs)
        
        # For each minimum, find closest needle using greedy assignment
        # (to avoid double-counting needles)
        min_distances = []
        used_needles = set()
        
        for i in range(minima_locs.shape[0]):
            # Find closest unused needle to this minimum
            min_dist = float('inf')
            best_needle_idx = -1
            
            for needle_idx in range(needles.shape[0]):
                if needle_idx not in used_needles:
                    dist = distances_to_minima[needle_idx, i].item()
                    if dist < min_dist:
                        min_dist = dist
                        best_needle_idx = needle_idx
            
            if best_needle_idx >= 0:
                min_distances.append(min_dist)
                used_needles.add(best_needle_idx)
            else:
                # No more needles available
                min_distances.append(float('inf'))
            
            status = "✅" if min_dist < 0.05 else "⚠️" if min_dist < 0.10 else "❌"
            print(f"  {status} Minimum {i+1:2d}: closest needle at distance {min_dist:.6f}")
        
        # Filter out infinite distances for statistics
        valid_distances = [d for d in min_distances if d != float('inf')]
        
        if valid_distances:
            print(f"\n  Mean distance: {np.mean(valid_distances):.6f}")
            print(f"  Max distance:  {np.max(valid_distances):.6f}")
            
            num_good = sum(1 for d in valid_distances if d < 0.05)
            num_ok = sum(1 for d in valid_distances if d < 0.10)
            print(f"  Within 0.05: {num_good}/{len(valid_distances)}")
            print(f"  Within 0.10: {num_ok}/{len(valid_distances)}")
            
            if num_ok == len(valid_distances):
                print("\n  🎉 SUCCESS! All minima found! 🎉")
            else:
                print(f"\n  ⚠️  Only {num_ok}/{len(valid_distances)} minima found within tolerance")
        else:
            print("\n  ❌ No valid needle assignments found")
    
    elif minima_locs is not None and needles.shape[0] == 0:
        print("No needles found to evaluate against true minima")
    
    # Generate plots if requested
    if plot and len(df) > 1:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Plot 1: Points sampled over time
        axes[0].plot(df['hours'], df['num_points_total'], 'b-', linewidth=2)
        axes[0].set_ylabel('Total Points Sampled', fontsize=12)
        axes[0].set_title(f'Trial {uuid} - Progress Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Needles found over time
        axes[1].plot(df['hours'], df['num_needles'], 'r-', linewidth=2, marker='o', markersize=6)
        if minima_locs is not None:
            target_minima = minima_locs.shape[0]
            axes[1].axhline(y=target_minima, color='g', linestyle='--', 
                          linewidth=2, label=f'Target ({target_minima} minima)')
        elif num_minima != 'unknown':
            axes[1].axhline(y=num_minima, color='g', linestyle='--', 
                          linewidth=2, label=f'Target ({num_minima} minima)')
        axes[1].set_ylabel('Needles Found', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Best value over time
        axes[2].plot(df['hours'], df['best_value'], 'g-', linewidth=2)
        axes[2].set_ylabel('Best Objective Value', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Input noise over time
        axes[3].plot(df['hours'], df['mean_distance'], 'purple', linewidth=2, label='Mean')
        axes[3].plot(df['hours'], df['median_distance'], 'orange', linewidth=2, label='Median')
        axes[3].set_ylabel('Input Noise (Euclidean)', fontsize=12)
        axes[3].set_xlabel('Time (hours)', fontsize=12)
        axes[3].legend(fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = trial_path / f'analysis_{uuid}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {plot_file}")
        
        # If needles and minima available, plot distance matrix
        if minima_locs is not None and needles.shape[0] > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            distances_matrix = torch.cdist(needles, minima_locs).numpy()
            
            im = ax.imshow(distances_matrix, cmap='RdYlGn_r', aspect='auto')
            ax.set_xlabel('True Minimum Index', fontsize=12)
            ax.set_ylabel('Needle Index', fontsize=12)
            ax.set_title(f'Distance Matrix: Needles vs True Minima', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Euclidean Distance', fontsize=12)
            
            # Add text annotations
            for i in range(distances_matrix.shape[0]):
                for j in range(distances_matrix.shape[1]):
                    text = ax.text(j, i, f'{distances_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            plt.tight_layout()
            
            distance_plot_file = trial_path / f'distances_{uuid}.png'
            plt.savefig(distance_plot_file, dpi=150, bbox_inches='tight')
            print(f"Saved distance matrix to {distance_plot_file}")
        
        # Generate convergence plot showing all points and needles
        print("Generating convergence plot...")
        generate_convergence_plot(uuid, trial_path, X_all_actual, Y_all, needles, needle_vals, minima_locs)
    
    print("\nAnalysis complete!")


def generate_convergence_plot(uuid, trial_path, X_all_actual, Y_all, needles, needle_vals, minima_locs=None):
    """
    Generate a convergence plot showing all sampled points and discovered needles.
    
    Args:
        uuid: Trial UUID
        trial_path: Path to trial directory
        X_all_actual: All sampled points (N, d)
        Y_all: All objective values (N, 1)
        needles: Discovered needle locations (M, d)
        needle_vals: Needle objective values (M, 1)
        minima_locs: True minima locations (optional)
    """
    try:
        # Convert to numpy for plotting
        X_np = X_all_actual.cpu().numpy()
        Y_np = Y_all.cpu().numpy().flatten()
        
        if needles.shape[0] > 0:
            needles_np = needles.cpu().numpy()
            needle_vals_np = needle_vals.cpu().numpy().flatten()
        else:
            needles_np = None
            needle_vals_np = None
        
        if minima_locs is not None:
            minima_np = minima_locs.cpu().numpy()
        else:
            minima_np = None
        
        # Create figure with subplots
        n_dims = X_np.shape[1]
        
        # For high dimensions, show first 2 dimensions and create a projection plot
        if n_dims > 2:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: First 2 dimensions
            ax1 = axes[0, 0]
            scatter = ax1.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap='viridis', 
                                alpha=0.6, s=20, label='All Points')
            
            if needles_np is not None:
                ax1.scatter(needles_np[:, 0], needles_np[:, 1], c='red', s=100, 
                           marker='*', edgecolors='black', linewidth=2, label='Needles')
            
            if minima_np is not None:
                ax1.scatter(minima_np[:, 0], minima_np[:, 1], c='orange', s=150, 
                           marker='X', edgecolors='black', linewidth=2, label='True Minima')
            
            ax1.set_xlabel('Dimension 1', fontsize=12)
            ax1.set_ylabel('Dimension 2', fontsize=12)
            ax1.set_title('First 2 Dimensions', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar1 = plt.colorbar(scatter, ax=ax1)
            cbar1.set_label('Objective Value', fontsize=10)
            
            # Plot 2: PCA projection (if we have enough points)
            if X_np.shape[0] > 10:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_np)
                
                ax2 = axes[0, 1]
                scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_np, cmap='viridis', 
                                     alpha=0.6, s=20, label='All Points')
                
                if needles_np is not None:
                    needles_pca = pca.transform(needles_np)
                    ax2.scatter(needles_pca[:, 0], needles_pca[:, 1], c='red', s=100, 
                               marker='*', edgecolors='black', linewidth=2, label='Needles')
                
                if minima_np is not None:
                    minima_pca = pca.transform(minima_np)
                    ax2.scatter(minima_pca[:, 0], minima_pca[:, 1], c='orange', s=150, 
                               marker='X', edgecolors='black', linewidth=2, label='True Minima')
                
                ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
                ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
                ax2.set_title('PCA Projection', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                cbar2 = plt.colorbar(scatter2, ax=ax2)
                cbar2.set_label('Objective Value', fontsize=10)
            
            # Plot 3: Objective value over sampling order
            ax3 = axes[1, 0]
            ax3.plot(range(len(Y_np)), Y_np, 'b-', alpha=0.7, linewidth=1, label='All Points')
            
            if needles_np is not None:
                # Find when each needle was discovered
                needle_indices = []
                for needle in needles_np:
                    # Find closest point in X_all_actual
                    distances = np.linalg.norm(X_np - needle, axis=1)
                    closest_idx = np.argmin(distances)
                    needle_indices.append(closest_idx)
                
                ax3.scatter(needle_indices, needle_vals_np, c='red', s=100, 
                           marker='*', edgecolors='black', linewidth=2, label='Needles')
            
            ax3.set_xlabel('Sampling Order', fontsize=12)
            ax3.set_ylabel('Objective Value', fontsize=12)
            ax3.set_title('Objective Value Over Time', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Distance from simplex center
            ax4 = axes[1, 1]
            simplex_center = np.ones(n_dims) / n_dims
            distances_from_center = np.linalg.norm(X_np - simplex_center, axis=1)
            
            scatter4 = ax4.scatter(distances_from_center, Y_np, c=Y_np, cmap='viridis', 
                                 alpha=0.6, s=20, label='All Points')
            
            if needles_np is not None:
                needle_distances = np.linalg.norm(needles_np - simplex_center, axis=1)
                ax4.scatter(needle_distances, needle_vals_np, c='red', s=100, 
                           marker='*', edgecolors='black', linewidth=2, label='Needles')
            
            if minima_np is not None:
                minima_distances = np.linalg.norm(minima_np - simplex_center, axis=1)
                # Get true objective values (approximate)
                ax4.scatter(minima_distances, [0] * len(minima_distances), c='orange', s=150, 
                           marker='X', edgecolors='black', linewidth=2, label='True Minima')
            
            ax4.set_xlabel('Distance from Simplex Center', fontsize=12)
            ax4.set_ylabel('Objective Value', fontsize=12)
            ax4.set_title('Objective vs Distance from Center', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            cbar4 = plt.colorbar(scatter4, ax=ax4)
            cbar4.set_label('Objective Value', fontsize=10)
            
        else:
            # For 2D case, create a simpler plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap='viridis', 
                               alpha=0.6, s=30, label='All Points')
            
            if needles_np is not None:
                ax.scatter(needles_np[:, 0], needles_np[:, 1], c='red', s=150, 
                          marker='*', edgecolors='black', linewidth=2, label='Needles')
            
            if minima_np is not None:
                ax.scatter(minima_np[:, 0], minima_np[:, 1], c='orange', s=200, 
                          marker='X', edgecolors='black', linewidth=2, label='True Minima')
            
            ax.set_xlabel('Dimension 1', fontsize=12)
            ax.set_ylabel('Dimension 2', fontsize=12)
            ax.set_title(f'Convergence Plot - Trial {uuid}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Objective Value', fontsize=12)
        
        plt.tight_layout()
        
        convergence_plot_file = trial_path / f'convergence_{uuid}.png'
        plt.savefig(convergence_plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {convergence_plot_file}")
        
        plt.close()  # Close to free memory
        
    except Exception as e:
        print(f"Error generating convergence plot: {e}")
        import traceback
        traceback.print_exc()


def compare_trials(trial_dir=None):
    """Compare multiple trial runs."""
    if trial_dir is None:
        # Auto-detect trial directory
        trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
        if not trial_dirs:
            print("No trial directories found")
            return
        
        # Use the first one by default
        trial_path = trial_dirs[0]
        print(f"Using trial directory: {trial_path.name}")
    else:
        trial_path = Path(trial_dir)
    
    checkpoints_dir = trial_path / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory found at {checkpoints_dir}")
        return
    
    all_runs = []
    
    for run_dir in sorted(checkpoints_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            uuid = run_dir.name.replace('run_', '')
            
            # Load final state
            with open(run_dir / 'current_state.txt', 'r') as f:
                final_state = f.read().strip()
            
            state_dir = run_dir / 'states' / final_state
            
            # Load stats
            with open(state_dir / 'stats.json', 'r') as f:
                stats = json.load(f)
            
            # Load tensors
            tensors = torch.load(state_dir / 'tensors.pt', map_location='cpu')
            
            all_runs.append({
                'uuid': uuid,
                'num_points': stats['num_points_total'],
                'num_needles': stats['num_needles'],
                'best_value': stats['best_value'],
                'needles': tensors['needles']
            })
    
    if not all_runs:
        print("No runs found")
        return
    
    print(f"\nComparison of {len(all_runs)} runs:")
    print("="*80)
    
    # Load minima
    minima_file = trial_path / 'minima_locations.pt'
    if minima_file.exists():
        minima_locs = torch.load(minima_file, map_location='cpu')
        
        for run in all_runs:
            # Evaluate using greedy assignment
            if run['needles'].shape[0] > 0:
                distances_matrix = torch.cdist(run['needles'], minima_locs)
                
                # Greedy assignment to avoid double-counting
                min_distances = []
                used_needles = set()
                
                for i in range(minima_locs.shape[0]):
                    min_dist = float('inf')
                    best_needle_idx = -1
                    
                    for needle_idx in range(run['needles'].shape[0]):
                        if needle_idx not in used_needles:
                            dist = distances_matrix[needle_idx, i].item()
                            if dist < min_dist:
                                min_dist = dist
                                best_needle_idx = needle_idx
                    
                    if best_needle_idx >= 0:
                        min_distances.append(min_dist)
                        used_needles.add(best_needle_idx)
                    else:
                        min_distances.append(float('inf'))
                
                # Filter out infinite distances
                valid_distances = [d for d in min_distances if d != float('inf')]
                mean_dist = np.mean(valid_distances) if valid_distances else float('inf')
                num_found = sum(1 for d in valid_distances if d < 0.10)
            else:
                mean_dist = float('inf')
                num_found = 0
            
            print(f"UUID: {run['uuid']}")
            print(f"  Points: {run['num_points']}, Needles: {run['num_needles']}")
            print(f"  Best value: {run['best_value']:.6f}")
            print(f"  Mean distance to minima: {mean_dist:.6f}")
            print(f"  Minima found (within 0.10): {num_found}/{minima_locs.shape[0]}")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            list_trials()
        elif sys.argv[1] == 'compare':
            compare_trials()
        else:
            uuid = sys.argv[1]
            trial_dir = sys.argv[2] if len(sys.argv) > 2 else None
            analyze_trial(uuid, trial_dir=trial_dir, plot=True)
    else:
        print("Usage:")
        print("  python analyze_24hour_trial.py list                    - List all trials")
        print("  python analyze_24hour_trial.py compare                 - Compare all trials")
        print("  python analyze_24hour_trial.py {UUID}                  - Analyze specific trial")
        print("  python analyze_24hour_trial.py {UUID} {trial_dir}      - Analyze trial in specific directory")
        print("\nExample:")
        print("  python analyze_24hour_trial.py a3f4")
        print("  python analyze_24hour_trial.py a3f4 trial_5minima_10d_24h")

