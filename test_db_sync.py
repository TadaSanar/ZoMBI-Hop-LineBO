#!/usr/bin/env python3
"""
Test script to verify database synchronization between objective_receiver and get_y_measurements.
"""

import time
import threading
import sqlite3
import os
from communication import _write_objective_row, _objective_db_lock, _objective_writing

def test_writer():
    """Simulate objective_receiver writing data"""
    for i in range(5):
        print(f"[Writer] Writing data batch {i+1}")
        test_values = [1.0 + i, 2.0 + i, 3.0 + i]
        _write_objective_row(test_values, "./sql/objective.db")
        time.sleep(1.0)

def test_reader():
    """Simulate get_y_measurements reading data"""
    for i in range(10):
        print(f"[Reader] Attempting to read data (attempt {i+1})")
        
        # Check if data is being written
        if _objective_writing:
            print(f"[Reader] Data is being written, waiting...")
            time.sleep(0.1)
            continue
        
        with _objective_db_lock:
            try:
                conn = sqlite3.connect("./sql/objective.db", timeout=10.0)
                cur = conn.cursor()
                
                cur.execute("SELECT * FROM objective")
                rows = cur.fetchall()
                
                if rows:
                    print(f"[Reader] Found {len(rows)} rows: {rows}")
                    
                    # Simulate clearing the table
                    cur.execute("DELETE FROM objective")
                    conn.commit()
                    print(f"[Reader] Cleared table")
                else:
                    print(f"[Reader] No data found")
                
                conn.close()
                
            except Exception as e:
                print(f"[Reader] Error: {e}")
        
        time.sleep(0.5)

def main():
    """Run the test"""
    print("🧪 Testing Database Synchronization")
    print("=" * 50)
    
    # Ensure SQL directory exists
    os.makedirs("./sql", exist_ok=True)
    
    # Start writer and reader threads
    writer_thread = threading.Thread(target=test_writer)
    reader_thread = threading.Thread(target=test_reader)
    
    writer_thread.start()
    reader_thread.start()
    
    # Wait for both threads to complete
    writer_thread.join()
    reader_thread.join()
    
    print("✅ Test completed")

if __name__ == "__main__":
    main() 