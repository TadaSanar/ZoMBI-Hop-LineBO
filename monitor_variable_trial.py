#!/usr/bin/env python3
"""
Real-time monitoring for ZoMBIHop trials with variable number of minima.
Displays live updates as the trial progresses.
"""

import time
import json
from pathlib import Path
import sys


def monitor_trial(uuid, trial_dir=None, refresh_seconds=30):
    """
    Monitor a running trial in real-time.
    
    Args:
        uuid: Trial UUID to monitor
        trial_dir: Trial directory (auto-detected if None)
        refresh_seconds: Update interval in seconds
    """
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
    
    print("="*80)
    print(f"MONITORING TRIAL: {uuid}")
    print("="*80)
    print(f"Refresh interval: {refresh_seconds} seconds")
    print("Press Ctrl+C to stop monitoring (trial will continue running)")
    print("="*80 + "\n")
    
    # Load metadata
    metadata_file = trial_path / 'trial_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        start_time = metadata.get('start_time', time.time())
        time_limit = metadata.get('time_limit_hours', 24.0)
    else:
        start_time = time.time()
        time_limit = 24.0
    
    # Load metadata to get trial parameters
    metadata_file = trial_path / 'trial_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        num_minima = metadata.get('num_minima', 'unknown')
        dimensions = metadata.get('dimensions', 'unknown')
        time_limit = metadata.get('time_limit_hours', 24.0)
    else:
        # Fallback to config file
        config_file = run_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            num_minima = config.get('d', 'unknown')
        else:
            num_minima = 'unknown'
        time_limit = 24.0
    
    last_state = None
    last_num_needles = 0
    
    try:
        while True:
            # Clear screen (works on Unix and Windows)
            print('\033[2J\033[H', end='')
            
            # Header
            print("="*80)
            print(f"TRIAL {uuid} - LIVE MONITORING")
            print("="*80)
            
            # Time info
            elapsed_hours = (time.time() - start_time) / 3600.0
            remaining_hours = time_limit - elapsed_hours
            percent_complete = (elapsed_hours / time_limit) * 100
            
            print(f"\n⏱️  TIME:")
            print(f"  Elapsed:   {elapsed_hours:.2f} hours")
            print(f"  Remaining: {remaining_hours:.2f} hours")
            print(f"  Progress:  {percent_complete:.1f}% of {time_limit:.1f} hour limit")
            
            # Progress bar
            bar_width = 50
            filled = int(bar_width * elapsed_hours / time_limit)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  [{bar}]")
            
            # Load current state
            current_state_file = run_dir / 'current_state.txt'
            if current_state_file.exists():
                with open(current_state_file, 'r') as f:
                    current_state = f.read().strip()
                
                state_dir = run_dir / 'states' / current_state
                
                if state_dir.exists():
                    # Load stats
                    stats_file = state_dir / 'stats.json'
                    if stats_file.exists():
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        print(f"\n📍 CURRENT STATE: {current_state}")
                        
                        # Parse activation/zoom/iteration from state name
                        if current_state.startswith('act'):
                            parts = current_state.split('_')
                            if len(parts) >= 3:
                                activation = parts[0].replace('act', '')
                                zoom = parts[1].replace('zoom', '')
                                iteration = parts[2].replace('iter', '')
                                print(f"  Activation: {activation}, Zoom: {zoom}, Iteration: {iteration}")
                        
                        print(f"\n📊 STATISTICS:")
                        print(f"  Total Points:   {stats.get('num_points_total', 0):,}")
                        if num_minima != 'unknown':
                            print(f"  Needles Found:  {stats.get('num_needles', 0)} / {num_minima}")
                        else:
                            print(f"  Needles Found:  {stats.get('num_needles', 0)}")
                        print(f"  Best Value:     {stats.get('best_value', 0):.6f}")
                        print(f"  Input Noise:    {stats.get('input_noise', 0):.6f}")
                        print(f"  Mean Distance:  {stats.get('mean_distance', 0):.6f}")
                        
                        # Needle progress bar
                        num_needles = stats.get('num_needles', 0)
                        if num_minima != 'unknown':
                            needle_bar_width = 50
                            needle_filled = int(needle_bar_width * num_needles / num_minima)
                            needle_bar = '🔹' * needle_filled + '□' * (needle_bar_width - needle_filled)
                            print(f"\n  Needles: [{needle_bar}] {num_needles}/{num_minima}")
                        else:
                            print(f"\n  Needles: {num_needles}")
                        
                        # Check if new needle found
                        if num_needles > last_num_needles:
                            print(f"\n  🎯 NEW NEEDLE FOUND! (#{num_needles})")
                            last_num_needles = num_needles
                        
                        # State change indicator
                        if current_state != last_state:
                            print(f"\n  ✨ State updated: {last_state or 'init'} → {current_state}")
                            last_state = current_state
                        
                        # Estimate completion
                        if num_needles > 0 and num_minima != 'unknown' and num_needles < num_minima:
                            needles_per_hour = num_needles / elapsed_hours if elapsed_hours > 0 else 0
                            if needles_per_hour > 0:
                                remaining_needles = num_minima - num_needles
                                est_hours = remaining_needles / needles_per_hour
                                print(f"\n  📈 Estimated time to find all minima: {est_hours:.1f} hours")
                                if est_hours > remaining_hours:
                                    print(f"     ⚠️  May not complete within time limit!")
            else:
                print("\n⚠️  No state file found - trial may not be running")
            
            # Check if trial is complete
            if elapsed_hours >= time_limit:
                print("\n" + "="*80)
                print("⏰ TIME LIMIT REACHED - Trial should be completing")
                print("="*80)
                break
            
            # Footer
            print("\n" + "="*80)
            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Next refresh in {refresh_seconds} seconds...")
            print("Press Ctrl+C to stop monitoring")
            print("="*80)
            
            # Wait before next update
            time.sleep(refresh_seconds)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Trial continues running.")
        print(f"Resume monitoring with: python monitor_trial.py {uuid}")


def list_running_trials(trial_dir=None):
    """List trials that appear to be active."""
    if trial_dir is None:
        # Auto-detect trial directories
        trial_dirs = [d for d in Path('.').iterdir() if d.is_dir() and d.name.startswith('trial_')]
        if not trial_dirs:
            print("No trial directories found")
            return []
        
        print("Found trial directories:")
        for i, td in enumerate(trial_dirs):
            print(f"  {i}: {td.name}")
        
        # Use the first one by default
        trial_path = trial_dirs[0]
        print(f"Using: {trial_path.name}")
    else:
        trial_path = Path(trial_dir)
    
    checkpoints_dir = trial_path / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory found at {checkpoints_dir}")
        return []
    
    # Load metadata to get start time
    metadata_file = trial_path / 'trial_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        start_time = metadata.get('start_time', 0)
        time_limit = metadata.get('time_limit_hours', 24.0)
    else:
        start_time = 0
        time_limit = 24.0
    
    print("\nSearching for active trials...")
    print("="*80)
    
    active_trials = []
    
    for run_dir in sorted(checkpoints_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            uuid = run_dir.name.replace('run_', '')
            
            # Load current state
            current_state_file = run_dir / 'current_state.txt'
            if not current_state_file.exists():
                continue
            
            with open(current_state_file, 'r') as f:
                current_state = f.read().strip()
            
            # Check if finished
            if current_state in ['final', 'timeout'] or 'finished' in current_state:
                status = "✅ Complete"
            else:
                # Check time
                if start_time > 0:
                    elapsed_hours = (time.time() - start_time) / 3600.0
                    if elapsed_hours < time_limit:
                        status = "🔄 Running"
                        active_trials.append(uuid)
                    else:
                        status = "⏰ Time limit reached"
                else:
                    status = "❓ Unknown"
            
            # Load stats
            state_dir = run_dir / 'states' / current_state
            if state_dir.exists():
                stats_file = state_dir / 'stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    print(f"{status} - UUID: {uuid}")
                    print(f"  State: {current_state}")
                    print(f"  Points: {stats.get('num_points_total', 0)}, Needles: {stats.get('num_needles', 0)}")
                    print()
    
    if active_trials:
        print(f"\nFound {len(active_trials)} potentially active trial(s)")
        print("Monitor with: python monitor_trial.py {UUID}")
    else:
        print("\nNo active trials found")
    
    print("="*80)
    
    return active_trials


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            list_running_trials()
        else:
            uuid = sys.argv[1]
            refresh = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            trial_dir = sys.argv[3] if len(sys.argv) > 3 else None
            monitor_trial(uuid, trial_dir=trial_dir, refresh_seconds=refresh)
    else:
        print("Usage:")
        print("  python monitor_trial.py list                              - List running trials")
        print("  python monitor_trial.py {UUID}                            - Monitor specific trial (30s refresh)")
        print("  python monitor_trial.py {UUID} {seconds}                  - Monitor with custom refresh")
        print("  python monitor_trial.py {UUID} {seconds} {trial_dir}      - Monitor trial in specific directory")
        print("\nExample:")
        print("  python monitor_trial.py a3f4")
        print("  python monitor_trial.py a3f4 10                           # Refresh every 10 seconds")
        print("  python monitor_trial.py a3f4 30 trial_5minima_10d_24h     # Monitor specific trial directory")

