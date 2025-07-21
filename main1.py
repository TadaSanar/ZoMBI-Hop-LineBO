import os
import multiprocessing
import initialize_databases
from communication import start_serial_dual_io_shared_port
from zombihop_linebo_v1 import run_zombi_main

def start_serial():
    start_serial_dual_io_shared_port(
        COM="COM5",
        baud=9600,
        obj_hz=10.0,
        comp_hz=2.0
    )

if __name__ == "__main__":

    initialize_databases.initialize_db()

    # On Windows ensure 'spawn' start method
    multiprocessing.set_start_method("spawn")

    # Process A: serial I/O
    p_serial = multiprocessing.Process(
        target=start_serial,
        name="SerialIO"
    )

    # Process B: ZoMBI‐Hop driver-
    p_zombi  = multiprocessing.Process(
        target=run_zombi_main,
        name="ZoMBI"
    )

    # Start them
    p_serial.start()
    p_zombi.start()

    # (Optionally) Wait for both to exit
    p_serial.join()
    p_zombi.join()
