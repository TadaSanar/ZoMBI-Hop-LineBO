#!/usr/bin/env python3
"""
Test script to verify objective table synchronization between objective_receiver and get_y_measurements.
"""

import time
import threading
import sqlite3
import os
import numpy as np
from communication import _write_objective_row, _objective_db_lock, _objective_writing

def simulate_objective_receiver():
    """Simulate objective_receiver writing data"""
    for i in range(3):
        print(f"[Simulated Receiver] Writing data batch {i+1}")
        test_values = [1.0 + i, 2.0 + i, 3.0 + i]
        _write_objective_row(test_values, "./sql/objective.db")
        time.sleep(2.0)  # Wait between writes

def simulate_get_y_measurements():
    """Simulate get_y_measurements reading data"""
    for i in range(5):
        print(f"[Simulated get_y_measurements] Attempt {i+1}")
        
        # Check if data is being written
        if _objective_writing:
            print(f"[Simulated get_y_measurements] Data is being written, waiting...")
            time.sleep(0.1)
            continue
        
        with _objective_db_lock:
            try:
                conn = sqlite3.connect("./sql/objective.db", timeout=10.0)
                cur = conn.cursor()
                
                cur.execute("SELECT * FROM objective")
                rows = cur.fetchall()
                
                if rows:
                    print(f"[Simulated get_y_measurements] Found {len(rows)} rows: {rows}")
                    
                    # Simulate processing the data
                    print(f"[Simulated get_y_measurements] Processing data...")
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Clear the table
                    cur.execute("DELETE FROM objective")
                    conn.commit()
                    print(f"[Simulated get_y_measurements] Cleared table")
                    
                    # Small delay after clearing
                    time.sleep(0.1)
                else:
                    print(f"[Simulated get_y_measurements] No data found")
                
                conn.close()
                
            except Exception as e:
                print(f"[Simulated get_y_measurements] Error: {e}")
        
        time.sleep(1.0)

def main():
    """Run the test"""
    print("🧪 Testing Objective Table Synchronization")
    print("=" * 50)
    
    # Ensure SQL directory exists
    os.makedirs("./sql", exist_ok=True)
    
    # Start both threads
    receiver_thread = threading.Thread(target=simulate_objective_receiver)
    reader_thread = threading.Thread(target=simulate_get_y_measurements)
    
    receiver_thread.start()
    reader_thread.start()
    
    # Wait for both threads to complete
    receiver_thread.join()
    reader_thread.join()
    
    print("✅ Test completed")

if __name__ == "__main__":
    main() 