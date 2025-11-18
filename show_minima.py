import sys
import torch as pytorch

m = pytorch.load('trial_24hour_10minima/minima_locations.pt', map_location='cpu')

print('='*80)
print('MINIMA LOCATIONS (10D Simplex)')
print('='*80)
print(f'Shape: {m.shape}')

for i in range(m.shape[0]):
    print(f'\nMinimum {i+1}:')
    coords = m[i].tolist()
    print(f'  [{", ".join([f"{x:.6f}" for x in coords])}]')
    print(f'  Sum: {m[i].sum().item():.10f}')

print('\n' + '='*80)
print('PAIRWISE DISTANCES')
print('='*80)

for i in range(m.shape[0]):
    for j in range(i+1, m.shape[0]):
        dist = pytorch.norm(m[i] - m[j]).item()
        print(f'Minimum {i+1} ↔ {j+1}: {dist:.6f}')

