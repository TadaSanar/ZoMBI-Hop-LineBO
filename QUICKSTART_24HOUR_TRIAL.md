# Quick Start Guide: 24-Hour ZoMBIHop Trial

## Step 1: Verify Requirements

Make sure you have:
- ✅ Python 3.8+
- ✅ PyTorch with CUDA support
- ✅ BoTorch and GPyTorch
- ✅ `test_functions_torch.py` in the same directory
- ✅ At least 50 GB free disk space
- ✅ CUDA-capable GPU

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check disk space
df -h .
```

## Step 2: Start the 24-Hour Trial

```bash
python test_24hour_10minima.py
```

You'll see output like:
```
================================================================================
24-HOUR ZOMBIHOP TRIAL: 10 Minima in 10D
================================================================================
Dimensions: 10
Number of minima: 10
Time limit: 24.0 hours
Device: cuda
================================================================================

Generating 10 distinguishable minima with min_distance=0.3...
  Minimum 1/10: min_dist to others = inf
  Minimum 2/10: min_dist to others = 0.4523
  ...

Starting new trial with UUID: a3f4

---------- Activation 1 ----------
Elapsed time: 0.00 / 24.00 hours
...
```

**⚠️ IMPORTANT: Note the UUID (e.g., `a3f4`) - you'll need it to resume!**

## Step 3: Monitor Progress (Optional)

In a **separate terminal**, run:

```bash
python monitor_trial.py a3f4
```

This will show a live dashboard updating every 30 seconds:
```
================================================================================
TRIAL a3f4 - LIVE MONITORING
================================================================================

⏱️  TIME:
  Elapsed:   2.34 hours
  Remaining: 21.66 hours
  Progress:  9.8% of 24.0 hour limit
  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

📍 CURRENT STATE: act1_zoom2_iter3
  Activation: 1, Zoom: 2, Iteration: 3

📊 STATISTICS:
  Total Points:   1,234
  Needles Found:  3 / 10
  Best Value:     12.345678
  Input Noise:    0.003456

  Needles: [🔹🔹🔹□□□□□□□] 3/10
```

Press `Ctrl+C` to stop monitoring (trial continues running).

## Step 4: Let It Run

The trial will run for 24 hours. You can:
- ✅ Close the terminal (if using `nohup` or `screen`)
- ✅ Check on it periodically with the monitor
- ✅ Let your computer sleep (not recommended)
- ✅ Stop it with `Ctrl+C` (state is saved, you can resume)

## Step 5: Resume if Interrupted

If the trial stops for any reason:

```bash
python test_24hour_10minima.py a3f4
```

It will continue from exactly where it left off!

## Step 6: Analyze Results

After 24 hours (or when it completes):

```bash
# View comprehensive analysis
python analyze_24hour_trial.py a3f4
```

This will show:
- Timeline of needles found
- Final distances to true minima
- Success/failure assessment
- Generate plots (`analysis_a3f4.png`, `distances_a3f4.png`)

## Quick Commands Reference

```bash
# Start new trial
python test_24hour_10minima.py

# Resume trial
python test_24hour_10minima.py {UUID}

# Monitor running trial
python monitor_trial.py {UUID}

# List all trials
python monitor_trial.py list
python analyze_24hour_trial.py list

# Analyze completed trial
python analyze_24hour_trial.py {UUID}

# Compare multiple trials
python analyze_24hour_trial.py compare
```

## Running in Background (Recommended)

### Using `screen` (Linux/Mac):
```bash
# Start screen session
screen -S zombihop_trial

# Run trial
python test_24hour_10minima.py

# Detach: Press Ctrl+A, then D

# Reattach later
screen -r zombihop_trial
```

### Using `tmux` (Linux/Mac):
```bash
# Start tmux session
tmux new -s zombihop_trial

# Run trial
python test_24hour_10minima.py

# Detach: Press Ctrl+B, then D

# Reattach later
tmux attach -t zombihop_trial
```

### Using `nohup` (Linux/Mac):
```bash
# Run in background, output to file
nohup python test_24hour_10minima.py > trial.log 2>&1 &

# Check output
tail -f trial.log

# Monitor in another terminal
python monitor_trial.py {UUID}
```

## Expected Timeline

Based on similar problems:

| Time | Expected Progress |
|------|-------------------|
| 0-2h | 2-4 needles found, ~500-1000 points sampled |
| 2-6h | 4-6 needles found, ~1500-2500 points |
| 6-12h | 6-8 needles found, ~2500-4000 points |
| 12-18h | 8-9 needles found, ~4000-5500 points |
| 18-24h | All 10 needles found (hopefully!), ~5000-6000 points |

## Success Criteria

✅ **Complete Success**: All 10 minima found within 0.05 distance
⚠️ **Partial Success**: All 10 minima found within 0.10 distance  
❌ **Needs Improvement**: Some minima not found or distance > 0.10

## Troubleshooting

### Trial crashes immediately
```bash
# Check imports
python -c "from test_functions_torch import MultiMinimaAckley"

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h .
```

### GPU out of memory
Edit `test_24hour_10minima.py` and reduce:
```python
max_gp_points=2000,  # was 3000
raw=5000,            # was 10000
n_restarts=50,       # was 100
```

### Trial seems stuck
```bash
# Check if still updating
ls -lht trial_24hour_10minima/checkpoints/run_*/states/ | head -20

# Monitor it
python monitor_trial.py {UUID}
```

### Want to stop and resume later
Just press `Ctrl+C` in the trial terminal. State is automatically saved.
Resume with: `python test_24hour_10minima.py {UUID}`

## Tips

1. **Start with a short test** (1-2 hours) to verify everything works:
   - Edit `test_24hour_10minima.py`
   - Change `TIME_LIMIT_HOURS = 2.0`
   - Run trial and verify it completes

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check checkpoint sizes**:
   ```bash
   du -sh trial_24hour_10minima/checkpoints/run_*/
   ```

4. **Save the UUID immediately** after starting:
   ```bash
   python test_24hour_10minima.py | tee trial_start.log
   grep "UUID:" trial_start.log
   ```

5. **Set up email notifications** (optional):
   ```bash
   # When trial completes, send email
   python test_24hour_10minima.py; echo "Trial complete" | mail -s "ZoMBIHop Done" you@email.com
   ```

## What Gets Created

```
trial_24hour_10minima/
├── minima_locations.pt              # Ground truth (10 minima)
├── trial_metadata.json              # Start time, config
├── results_a3f4.json                # Final results
├── analysis_a3f4.png                # Progress plots
├── distances_a3f4.png               # Distance heatmap
└── checkpoints/
    └── run_a3f4/
        ├── config.json
        ├── current_state.txt
        └── states/
            ├── init/
            ├── act0_zoom0_iter0/
            ├── act0_zoom0_iter1/
            ├── ...
            ├── act0_zoom0_iter5_needle/
            └── final/
                ├── tensors.pt        # All optimization data
                ├── needles_results.json
                ├── tracking.json
                ├── stats.json
                ├── all_points.csv    # Human-readable data
```

## After Completion

1. **Analyze results**:
   ```bash
   python analyze_24hour_trial.py a3f4
   ```

2. **Check success**:
   ```bash
   cat trial_24hour_10minima/results_a3f4.json
   ```

3. **View plots**:
   ```bash
   open trial_24hour_10minima/analysis_a3f4.png
   open trial_24hour_10minima/distances_a3f4.png
   ```

4. **Export data for further analysis**:
   ```bash
   # CSV files are in each state directory
   head trial_24hour_10minima/checkpoints/run_a3f4/states/final/all_points.csv
   ```

## Questions?

- Check `24HOUR_TEST_README.md` for detailed documentation
- Check `CHECKPOINTING_GUIDE.md` for checkpoint system details
- Check `IMPLEMENTATION_SUMMARY.md` for technical overview

## Ready to Start?

```bash
# Let's go!
python test_24hour_10minima.py
```

Good luck! 🚀

