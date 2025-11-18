import os
import multiprocessing
import signal
import time
import sys
from initialize_databases import initialize_db
from communication import start_serial_dual_io_shared_port
from zombihop_linebo_v1 import run_zombi_main
import communication  # <-- Add this import for explicit resets

def start_serial():
    """Start the serial communication process"""
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

def start_zombi():
    """Start the ZoMBI-Hop optimization process"""
    try:
        # Wait a moment for serial process to initialize
        time.sleep(2)
        run_zombi_main()
    except Exception as e:
        print(f"[ZoMBI Process] Error: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[Main] Received signal {signum}, shutting down processes...")
    sys.exit(0)

if __name__ == "__main__":
    # Hard reset all DBs and communication state
    initialize_db()
    communication.reset_objective()
    communication.reset_compositions()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize databases
    try:
        initialize_db()
        print("[Main] Databases initialized successfully")
    except Exception as e:
        print(f"[Main] Error initializing databases: {e}")
        sys.exit(1)

    # On Windows ensure 'spawn' start method
    multiprocessing.set_start_method("spawn", force=True)

    # Process A: serial I/O
    p_serial = multiprocessing.Process(
        target=start_serial,
        name="SerialIO"
    )

    # Process B: ZoMBI‐Hop driver
    p_zombi = multiprocessing.Process(
        target=start_zombi,
        name="ZoMBI"
    )

    try:
        # Start serial process first
        print("[Main] Starting serial communication process...")
        p_serial.start()
        
        # Wait a moment for serial to initialize
        time.sleep(3)
        
        # Check if serial process is still alive
        if not p_serial.is_alive():
            print("[Main] Serial process failed to start or died immediately")
            sys.exit(1)
        
        # Start ZoMBI process
        print("[Main] Starting ZoMBI-Hop optimization process...")
        p_zombi.start()
        
        # Monitor both processes
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
        # Cleanup
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
