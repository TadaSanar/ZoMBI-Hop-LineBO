"""
ZoMBI-Hop v2 Main Launcher with Database Communication

This script starts two parallel processes:
1. Serial communication process (Archerfish interface)
2. ZoMBI-Hop v2 optimization process

Usage:
    python -m scripts.main              # Start new trial
    python -m scripts.main <uuid>       # Resume trial from UUID
    python -m scripts.main list         # List available trials

Examples:
    python -m scripts.main              # New trial
    python -m scripts.main a2fe         # Resume trial with UUID 'a2fe'
    python -m scripts.main list         # Show all available trials

The trial UUID will be printed when starting. Use Ctrl+C to stop gracefully.
Checkpoints are saved automatically to: actual_runs/checkpoints/run_<uuid>/
"""

import os
import multiprocessing
import signal
import time
import sys
from pathlib import Path

from scripts.initialize_databases import initialize_db
from scripts.communication import start_serial_dual_io_shared_port
from scripts import communication
from scripts.run_zombi_main import run_zombi_main


def list_runs_and_exit():
    base = Path('actual_runs')
    print("Available trials in 'actual_runs':")
    print("="*80)
    if not base.exists():
        print("No trials found.")
        return
    trials = [d for d in base.iterdir() if d.is_dir()]
    if not trials:
        print("No trials found.")
    else:
        for td in sorted(trials):
            print(f"\nTrial directory: {td.name}")
            checkpoints_dir = td / 'checkpoints'
            if checkpoints_dir.exists():
                for run_dir in sorted(checkpoints_dir.iterdir()):
                    if run_dir.is_dir() and run_dir.name.startswith('run_'):
                        uuid = run_dir.name.replace('run_', '')
                        meta = td / 'trial_metadata.json'
                        if meta.exists():
                            import json
                            with open(meta, 'r') as f:
                                m = json.load(f)
                            print(f"  UUID: {uuid} ({m.get('num_minima','?')} minima, {m.get('dimensions','?')}D, {m.get('time_limit_hours','?')}h)")
                        else:
                            print(f"  UUID: {uuid}")
            else:
                print("  No checkpoints found")


def start_serial():
    try:
        start_serial_dual_io_shared_port(
            COM="COM5",
            baud=9600,
            obj_hz=1.0,
            comp_hz=1.0,
            chaos=True
        )
    except Exception as e:
        print(f"[Serial Process] Error: {e}")
        sys.exit(1)


def start_zombi(resume_uuid=None):
    try:
        time.sleep(2)
        if resume_uuid:
            print(f"[ZoMBI Process] Resuming ZoMBI-Hop v2 with UUID: {resume_uuid}...")
        else:
            print("[ZoMBI Process] Starting ZoMBI-Hop v2 (DB-driven)...")
        run_zombi_main(resume_uuid=resume_uuid)
    except Exception as e:
        print(f"[ZoMBI Process] Error: {e}")
        sys.exit(1)


def signal_handler(signum, frame):
    print(f"\n[Main] Received signal {signum}, shutting down processes...")
    sys.exit(0)


def main():
    Path('actual_runs').mkdir(exist_ok=True)

    # Parse command line arguments
    resume_uuid = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check for 'list' command
        if arg.lower() == 'list':
            list_runs_and_exit()
            sys.exit(0)

        # Otherwise treat as resume UUID
        resume_uuid = arg
        print(f"[Main2] Resume UUID provided: {resume_uuid}")

    # Hard reset all DBs and communication state ONLY if starting new trial
    if resume_uuid is None:
        initialize_db()
        communication.reset_objective()
        communication.reset_compositions()
        print("[Main] Database reset complete (new trial)")
    else:
        print("[Main] Skipping database reset (resuming trial)")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        initialize_db()
        print("[Main] Databases initialized successfully")
    except Exception as e:
        print(f"[Main] Error initializing databases: {e}")
        sys.exit(1)

    multiprocessing.set_start_method("spawn", force=True)

    p_serial = multiprocessing.Process(target=start_serial, name="SerialIO")
    p_zombi = multiprocessing.Process(target=start_zombi, args=(resume_uuid,), name="ZoMBI")

    try:
        print("[Main] Starting serial communication process...")
        p_serial.start()
        time.sleep(3)
        if not p_serial.is_alive():
            print("[Main] Serial process failed to start or died immediately")
            sys.exit(1)

        print("[Main] Starting ZoMBI-Hop optimization process...")
        p_zombi.start()

        while True:
            if not p_serial.is_alive():
                print("[Main] Serial process died unexpectedly")
                if p_zombi.is_alive():
                    print("[Main] Terminating ZoMBI process...")
                    p_zombi.terminate()
                    p_zombi.join(timeout=5)
                break

            if not p_zombi.is_alive():
                print("[Main] ZoMBI process completed or died")
                if p_serial.is_alive():
                    print("[Main] Terminating serial process...")
                    p_serial.terminate()
                    p_serial.join(timeout=5)
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received, shutting down...")
    except Exception as e:
        print(f"[Main] Unexpected error: {e}")
    finally:
        print("[Main] Cleaning up processes...")

        if p_serial.is_alive():
            print("[Main] Terminating serial process...")
            p_serial.terminate()
            p_serial.join(timeout=5)
            if p_serial.is_alive():
                print("[Main] Force killing serial process...")
                p_serial.kill()

        try:
            import serial
            port_name = "COM5"
            try:
                s = serial.Serial(port_name)
                if s.is_open:
                    print(f"[Main] Closing serial port {port_name}...")
                    s.close()
            except Exception as e:
                print(f"[Main] Could not close port {port_name}: {e}")
        except Exception as e:
            print(f"[Main] Serial port cleanup skipped or failed: {e}")

        if p_zombi.is_alive():
            print("[Main] Terminating ZoMBI process...")
            p_zombi.terminate()
            p_zombi.join(timeout=5)
            if p_zombi.is_alive():
                print("[Main] Force killing ZoMBI process...")
                p_zombi.kill()

        print("[Main] Cleanup complete")


if __name__ == "__main__":
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        print("\nCurrent configuration:")
        print(f"  Serial port: COM5")
        print(f"  Checkpoint directory: actual_runs/checkpoints/")
        print(f"  Device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
        sys.exit(0)

    main()
