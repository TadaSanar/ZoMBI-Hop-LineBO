# Checkpoint Optimization: Rolling Window System

## Overview

The checkpoint system has been optimized to save disk space while maintaining full resumability. The new system uses a **rolling window** approach that keeps:

1. **Last 50 iterations** (configurable)
2. **All permanent checkpoints** (zoom ends, needle discoveries, key milestones)

This dramatically reduces disk usage while preserving all important optimization milestones.

## How It Works

### Checkpoint Types

**Temporary Checkpoints (Rolling):**
- Regular iterations: `act{A}_zoom{Z}_iter{I}`
- Only the last 50 are kept
- Automatically deleted when more than 50 iterations exist
- Can still resume from any of the last 50 iterations

**Permanent Checkpoints (Never Deleted):**
- Initial state: `init`
- Needle discoveries: `act{A}_zoom{Z}_iter{I}_needle`
- Failed iterations: `act{A}_zoom{Z}_iter{I}_failed`
- Zoom completions: `act{A}_zoom{Z}_complete`
- Optimization finished: `act{A}_zoom{Z}_finished`
- Timeouts: `act{A}_timeout` or `act{A}_zoom{Z}_iter{I}_timeout`
- Final state: `final`

### Automatic Cleanup

After each save, the system:
1. Checks if there are more than 50 checkpoints
2. Identifies which to keep:
   - Last 50 iterations
   - All permanent checkpoints
3. Deletes old non-permanent checkpoints
4. Prints cleanup summary

## Example Timeline

```
Iteration 1-50:    Saved [all kept]
Iteration 51:      Saved, deleted iteration 1
Iteration 52:      Saved, deleted iteration 2
...
Iteration 100:     Saved, deleted iteration 50
Needle found:      Saved iteration 100 as PERMANENT (never deleted)
Iteration 101:     Saved, deleted iteration 51
...
Zoom complete:     Saved iteration 150 as PERMANENT
...
Final:             Saved as PERMANENT
```

**Result:** Only ~50-70 checkpoints instead of hundreds, but all key moments preserved!

## Disk Space Savings

### Before Optimization
- 24-hour run with 500 iterations
- ~500 checkpoints × 50 MB each
- **Total: ~25 GB**

### After Optimization
- Same 500 iterations
- ~50 recent + ~15 permanent checkpoints
- **Total: ~3.25 GB**

**Space saved: ~87%** 🎉

## Configuration

You can adjust the rolling window size when initializing ZoMBIHop:

```python
optimizer = ZoMBIHop(
    ...
)

# Change rolling window size (default is 50)
optimizer.max_recent_iterations = 100  # Keep last 100 iterations
```

Or modify directly in the code:

```python
# In __init__ method
self.max_recent_iterations = 50  # Change this value
```

## What Gets Saved

Each checkpoint still contains:
- All tensors (X_all, Y_all, needles, distances, masks, etc.)
- Needles results JSON
- Tracking info (activation/zoom/iteration)
- Statistics (best value, noise levels, distances)
- CSV file (all points and distances)

Nothing is removed from individual checkpoints - we just save fewer of them!

## Resumability

You can still resume from:
- ✅ Any of the last 50 iterations
- ✅ Any needle discovery
- ✅ Any zoom completion
- ✅ Any activation end
- ✅ Final state
- ✅ Timeout states

You **cannot** resume from:
- ❌ Iterations older than the last 50 (unless they were permanent)

This is usually fine because if something fails, it's typically recent.

## Console Output

When saving checkpoints, you'll see:

```
Saved state to .../states/act1_zoom0_iter5
Saved state to .../states/act1_zoom0_iter6
...
Saved state to .../states/act1_zoom0_iter55
  Cleaned up 5 old checkpoints (keeping last 50 + permanents)
...
Saved state to .../states/act1_zoom0_iter100_needle [PERMANENT]
```

Permanent checkpoints are clearly marked with `[PERMANENT]`.

## Monitoring Disk Usage

To see current disk usage:

```bash
# Total size of checkpoints
du -sh trial_24hour_10minima/checkpoints/

# Number of checkpoints
ls trial_24hour_10minima/checkpoints/run_*/states/ | wc -l

# Size of individual checkpoints
du -sh trial_24hour_10minima/checkpoints/run_*/states/*
```

## Recovery After Resuming

When you resume from a checkpoint, the system:
1. Loads all existing states from disk
2. Reconstructs checkpoint history
3. Automatically marks permanent checkpoints based on their names
4. Continues cleanup from that point

Example:

```python
# Resume from UUID
optimizer = ZoMBIHop(
    objective=my_objective,
    bounds=None,
    X_init_actual=None,
    X_init_expected=None,
    Y_init=None,
    run_uuid="a3f4"
)

# Output:
# Loaded state: activation=1, zoom=2, iteration=7
# Checkpoint history: 65 states found
# (continues optimization and cleanup)
```

## Benefits

1. **87% less disk space** - Run longer trials without filling disk
2. **Faster saves** - Less to clean up means quicker checkpointing
3. **Still fully resumable** - Can resume from recent states or key milestones
4. **Automatic management** - No manual cleanup needed
5. **Preserves history** - All important moments are kept forever

## Technical Details

### Checkpoint Tracking

The system maintains a list of all checkpoints:

```python
self.checkpoint_history = [
    ("init", True),                           # Permanent
    ("act0_zoom0_iter0", False),             # Temporary
    ("act0_zoom0_iter1", False),             # Temporary
    ...
    ("act0_zoom0_iter100_needle", True),     # Permanent
    ...
]
```

### Cleanup Algorithm

```python
def _cleanup_old_checkpoints(self):
    # Keep permanent checkpoints
    keep = {label for label, perm in history if perm}
    
    # Keep last N iterations
    keep.update([label for label, _ in history[-N:]])
    
    # Delete everything else
    for label in history:
        if label not in keep:
            delete(label)
```

### Space Complexity

- **Memory:** O(1) - Only stores checkpoint labels, not data
- **Disk:** O(N + P) where N=window size, P=permanent checkpoints
- **Time:** O(H) per cleanup where H=history length (very fast)

## Troubleshooting

**Q: I need more than 50 recent iterations**

A: Increase `max_recent_iterations`:
```python
optimizer.max_recent_iterations = 100
```

**Q: Can I make fewer checkpoints permanent?**

A: Yes, modify the `is_permanent` flags in the `run()` method. For example, to not save zoom completions as permanent:

```python
# Change from:
self._save_state(iteration_label, is_permanent=True)
# To:
self._save_state(iteration_label, is_permanent=False)
```

**Q: What if I want all checkpoints?**

A: Set a very large window:
```python
optimizer.max_recent_iterations = 1000000  # Effectively infinite
```

**Q: Disk is still filling up**

A: Check if CSV files are the issue. You can disable them by commenting out the CSV generation in `_save_state()`, or reduce the window size further:
```python
optimizer.max_recent_iterations = 25  # Keep only last 25
```

## Comparison with Other Approaches

| Approach | Disk Usage | Detail | Complexity |
|----------|-----------|--------|------------|
| Save Everything | Very High | Very High | Simple |
| Save Every Nth | Medium | Medium | Simple |
| Incremental | Low | High | Complex |
| **Rolling Window** | **Low** | **High** | **Medium** |
| Minimal Only | Very Low | Low | Simple |

The rolling window approach provides the best balance of space efficiency and detail preservation.

## Future Enhancements

Possible improvements:
- [ ] Compress old checkpoints instead of deleting
- [ ] Configurable permanent checkpoint types
- [ ] Disk space monitoring with automatic cleanup
- [ ] Archive mode for long-term storage

