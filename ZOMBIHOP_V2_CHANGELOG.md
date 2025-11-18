# ZoMBI-Hop v2 Changelog

## Updates based on user requirements

### 1. 4-Digit UUIDs
- **Changed**: UUIDs are now 4-digit hexadecimal codes (e.g., `a2fe`, `f358`)
- **Previous**: Full UUID format (e.g., `a2fe1234-5678-90ab-cdef-1234567890ab`)
- **Implementation**: New `generate_short_uuid()` function generates 4-digit hex strings
- **Example**: `run_a2fe`, `run_f358`, `run_9722`

### 2. Unlimited by Default
- **Changed**: New trials run with unlimited time and activations by default
- **Previous**: Default was 5 activations with no time limit
- **Implementation**: 
  - `max_activations` parameter default changed from `5` to `None` (unlimited)
  - Command-line `--activations` default changed from `5` to `None`
  - `time_limit_hours` remains `None` (unlimited) by default
  
### 3. Resume Support
- **Feature**: Can resume any trial by passing its 4-digit UUID
- **Usage**: `python zombihop_linebo_v2.py --resume a2fe`
- **Checkpoint Location**: `./checkpoints/run_<uuid>/`

## Usage Examples

### Start New Trial (Unlimited)
```bash
# Run until manually stopped
python zombihop_linebo_v2.py
```

```python
# Programmatic: unlimited
results = run_zombi_main()
```

### Start New Trial (With Limits)
```bash
# Specify limits
python zombihop_linebo_v2.py --activations 5 --time-limit 24
```

```python
# Programmatic: with limits
results = run_zombi_main(max_activations=5, time_limit_hours=24)
```

### Resume Trial
```bash
# Resume using 4-digit UUID
python zombihop_linebo_v2.py --resume a2fe
```

```python
# Programmatic: resume
results = run_zombi_main(resume_uuid='a2fe')
```

## Default Parameters

```python
def run_zombi_main(
    resume_uuid: Optional[str] = None,      # None = new trial, "a2fe" = resume
    max_activations: Optional[int] = None,  # None = unlimited, int = limit
    time_limit_hours: Optional[float] = None # None = unlimited, float = limit
):
    ...
```

## Command-Line Arguments

```bash
python zombihop_linebo_v2.py [OPTIONS]

Options:
  --resume TEXT          UUID of trial to resume (4-digit hex)
  --activations INTEGER  Maximum number of activations (default: unlimited)
  --time-limit FLOAT     Time limit in hours (default: unlimited)
```

## Finding Your Trial UUID

When you start a new trial:
```
✅ Starting new trial with UUID: a2fe
```

Or check checkpoint directories:
```bash
ls checkpoints/
# Output: run_a2fe  run_f358  run_9722
```

Or check config file:
```bash
cat checkpoints/run_a2fe/config.json
```

## Migration from v1

### Old Code (v1)
```python
# v1: Always required specifying activations
optimizer.run_zombi_hop(num_activations=5)
```

### New Code (v2)
```python
# v2: Default is unlimited
zombihop.run()  # Runs forever until stopped

# Or specify limits
zombihop.run(max_activations=5, time_limit_hours=24)
```

## Benefits

1. **Simpler Start**: Just run `python zombihop_linebo_v2.py` - no parameters needed
2. **Easy Resume**: Use short 4-digit codes instead of long UUIDs
3. **Flexible**: Can run unlimited or set specific limits
4. **Safe**: Automatically checkpoints, so you can stop/resume anytime

## Examples from Your Codebase

Looking at your trial directories:
- `trial_10minima_10d_24h/checkpoints/run_a2fe/` ✅ 4-digit
- `trial_10minima_10d_24h/checkpoints/run_f358/` ✅ 4-digit
- `trial_24hour_10minima/checkpoints/run_7314/` ✅ 4-digit

These all use the 4-digit UUID pattern that v2 now generates by default!



