"""
Quick script to view the minima locations used in the 24-hour trial.
"""

import torch
from pathlib import Path

# Load minima
minima_file = Path('trial_24hour_10minima/minima_locations.pt')

if not minima_file.exists():
    print(f"Minima file not found at {minima_file}")
    print("Make sure you're in the correct directory and the trial has been started.")
else:
    minima = torch.load(minima_file, map_location='cpu')
    
    print("="*80)
    print("MINIMA LOCATIONS FOR 24-HOUR TRIAL")
    print("="*80)
    print(f"\nShape: {minima.shape}")
    print(f"Number of minima: {minima.shape[0]}")
    print(f"Dimensions: {minima.shape[1]}")
    print("\n" + "="*80)
    
    # Print each minimum
    for i in range(minima.shape[0]):
        print(f"\nMinimum {i+1}:")
        print(f"  Coordinates: {minima[i].tolist()}")
        print(f"  Sum (should be ~1.0): {minima[i].sum().item():.6f}")
    
    # Calculate and display pairwise distances
    print("\n" + "="*80)
    print("PAIRWISE DISTANCES BETWEEN MINIMA")
    print("="*80)
    
    for i in range(minima.shape[0]):
        for j in range(i+1, minima.shape[0]):
            dist = torch.norm(minima[i] - minima[j]).item()
            print(f"  Distance between minimum {i+1:2d} and {j+1:2d}: {dist:.6f}")
    
    # Find minimum and maximum pairwise distances
    distances = []
    for i in range(minima.shape[0]):
        for j in range(i+1, minima.shape[0]):
            distances.append(torch.norm(minima[i] - minima[j]).item())
    
    print("\n" + "="*80)
    print("DISTANCE STATISTICS")
    print("="*80)
    print(f"  Minimum distance: {min(distances):.6f}")
    print(f"  Maximum distance: {max(distances):.6f}")
    print(f"  Mean distance:    {sum(distances)/len(distances):.6f}")
    print("\n" + "="*80)
    
    # Optionally save as CSV for easy viewing
    import csv
    csv_file = 'trial_24hour_10minima/minima_locations.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['minimum_id'] + [f'dim_{i}' for i in range(minima.shape[1])] + ['sum']
        writer.writerow(header)
        
        for i in range(minima.shape[0]):
            row = [i+1] + minima[i].tolist() + [minima[i].sum().item()]
            writer.writerow(row)
    
    print(f"\nMinima also saved to: {csv_file}")

