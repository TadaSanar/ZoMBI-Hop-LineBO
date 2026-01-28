# Main2.py Usage Guide - ZoMBI-Hop v2 with Database Communication

## Overview

`main2.py` is the main launcher for ZoMBI-Hop v2 with real-time database communication to the Archerfish system. It runs two parallel processes:

1. **Serial Communication Process**: Handles hardware interface (COM5)
2. **ZoMBI-Hop v2 Process**: Runs optimization using `zombihop_linebo_v2.py`

## Quick Start

### Starting a New Trial

```bash
python main2.py
```

This will:
- Reset all databases
- Initialize a new optimization trial
- Print a 4-digit UUID (e.g., `a2fe`)
- Run indefinitely until stopped with Ctrl+C

### Resuming a Trial

```bash
python main2.py a2fe
```

Replace `a2fe` with your trial's 4-digit UUID. This will:
- Load the checkpoint from `actual_runs/checkpoints/run_a2fe/`
- Resume optimization from the saved state
- Continue indefinitely until stopped

### Listing Available Trials

```bash
python main2.py list
```

Shows all available trials with their UUIDs and metadata.

### Getting Help

```bash
python main2.py --help
```

## Usage Comparison

### main1.py (v1 - Old)
```bash
python main1.py                    # New trial
# No built-in resume support
```

### main2.py (v2 - New)
```bash
python main2.py                    # New trial
python main2.py a2fe               # Resume trial
python main2.py list               # List trials
```

## Process Management

### Starting
When you run `main2.py`:
1. Serial process starts first (3 second delay)
2. ZoMBI process starts second
3. Both processes run in parallel

### Stopping
Press `Ctrl+C` to stop gracefully:
```
[Main2] KeyboardInterrupt received, shutting down...
[Main2] Cleaning up processes...
[Main2] Terminating serial process...
[Main2] Terminating ZoMBI process...
[Main2] Cleanup complete
```

The trial UUID will be printed so you can resume later.

### Automatic Cleanup
If a process dies unexpectedly:
- The other process is automatically terminated
- Serial port is properly closed
- You can safely restart with the same UUID

## Checkpoint Structure

```
actual_runs/
└── checkpoints/
    ├── run_a2fe/
    │   ├── config.json
    │   ├── current_state.txt
    │   └── states/
    │       ├── init/
    │       ├── act0_zoom0_iter0/
    │       ├── act0_zoom0_iter1/
    │       └── ...
    ├── run_f358/
    └── run_9722/
```

## Configuration

### Serial Port Settings
Located in `main2.py`:
```python
start_serial_dual_io_shared_port(
    COM="COM5",           # Serial port
    baud=9600,            # Baud rate
    obj_hz=1.0,           # Objective polling rate
    comp_hz=1.0,          # Composition polling rate
    chaos=True            # Chaos mode
)
```

### ZoMBIHop Parameters
Located in `zombihop_linebo_v2.py`:
```python
OPTIMIZING_DIMS = [0, 1, 8]      # Which dimensions to optimize

optimizer = ZoMBIHop(
    penalization_threshold=1e-3,
    improvement_threshold_noise_mult=1.5,
    input_noise_threshold_mult=3.4,
    n_consecutive_no_improvements=5,
    top_m_points=3,
    max_zooms=2,
    max_iterations=10,
    n_restarts=50,
    raw=5000,
    penalty_num_directions=100,
    penalty_max_radius=0.3,
    penalty_radius_step=0.01,
    max_gp_points=3000,
)
```

## Database Files

The system uses three SQLite databases in `./sql/`:

1. **objective.db**: Main objective values from hardware
2. **compositions.db**: Composition data sent to hardware  
3. **objective_memory.db**: Memory cache for objectives

### Handshake Protocol
- **New Trial**: Processes any available data immediately
- **After Initialization**: Waits for `new_objective_available` flag before reading

## Differences from main1.py / v1

| Feature | main1.py (v1) | main2.py (v2) |
|---------|---------------|---------------|
| Resume support | ❌ No | ✅ Yes (4-digit UUID) |
| List trials | ❌ No | ✅ Yes (`list` command) |
| Checkpoint system | ❌ No | ✅ Automatic |
| Database reset | Always | Only for new trials |
| UUID format | N/A | 4-digit hex (e.g., `a2fe`) |

## Troubleshooting

### Serial Port Issues

**Problem**: Serial port won't open
```
[Serial Process] Error: could not open port 'COM5'
```

**Solution**:
1. Check device manager - ensure device is on COM5
2. Close other programs using the port
3. Update COM port in `main2.py` if using different port

### Resume Not Working

**Problem**: Can't find UUID
```
❌ Error: Checkpoint not found for UUID: a2fe
```

**Solution**:
1. Run `python main2.py list` to see available UUIDs
2. Check `actual_runs/checkpoints/` directory exists
3. Ensure you're using the correct 4-digit UUID

### Process Won't Stop

**Problem**: Process hangs after Ctrl+C

**Solution**:
1. Wait 5 seconds for graceful shutdown
2. If still hanging, close terminal window
3. Manually kill processes in Task Manager (Windows) or `ps`/`kill` (Linux)

### Database Lock Errors

**Problem**: Database is locked
```
[get_y_measurements] database is locked
```

**Solution**:
1. Wait - usually resolves automatically
2. Check no other processes accessing databases
3. Restart both processes if persistent

## Advanced Usage

### Running on Different Serial Port

Edit `main2.py`:
```python
start_serial_dual_io_shared_port(
    COM="COM3",  # Change to your port
    baud=9600,
    obj_hz=1.0,
    comp_hz=1.0,
    chaos=True
)
```

### Changing Optimization Dimensions

Edit `zombihop_linebo_v2.py`:
```python
OPTIMIZING_DIMS = [0, 2, 5]  # Optimize different dimensions
```

### Running Without Hardware (Debug Mode)

For debugging without hardware, you can modify the objective functions to use synthetic data. See `test_24hour_variable_minima.py` for examples.

## Example Session

```bash
# Start new trial
$ python main2.py
[Main2] Database reset complete (new trial)
[Main2] Databases initialized successfully
[Main2] Starting serial communication process...
[Main2] Starting ZoMBI-Hop optimization process...
[ZoMBI Process] Starting ZoMBI-Hop v2 (DB-driven)...
✅ Starting new trial with UUID: a2fe
================================================================================
STARTING OPTIMIZATION
================================================================================

# ... optimization runs ...
# Press Ctrl+C to stop

^C
[Main2] KeyboardInterrupt received, shutting down...
Trial UUID: a2fe
Resume with: python main2.py a2fe

# Later, resume the trial
$ python main2.py a2fe
[Main2] Resume UUID provided: a2fe
[Main2] Skipping database reset (resuming trial)
[ZoMBI Process] Resuming ZoMBI-Hop v2 with UUID: a2fe...
✅ Resumed from activation=2, zoom=1, iteration=5

# ... optimization continues ...
```

## Monitoring Progress

While running, you can monitor in another terminal:

```bash
# Watch checkpoints being created
ls -lt actual_runs/checkpoints/run_a2fe/states/

# Check current state
cat actual_runs/checkpoints/run_a2fe/current_state.txt

# View configuration
cat actual_runs/checkpoints/run_a2fe/config.json
```

## Best Practices

1. **Always let processes shut down gracefully** - Press Ctrl+C once and wait
2. **Note the UUID when starting** - You'll need it to resume
3. **Check database health** - If seeing lock errors, restart processes
4. **Monitor disk space** - Checkpoints accumulate over time
5. **Use `list` command** - Keep track of all your trials

## See Also

- `zombihop_linebo_v2.py` - Core optimization implementation
- `zombihop_linebo_final.py` - Base ZoMBIHop class with checkpointing
- `communication.py` - Database communication functions
- `test_24hour_variable_minima.py` - Testing without hardware

