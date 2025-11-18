#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot communication issues between processes.
This script helps identify common problems with serial communication and database access.
"""

import os
import sqlite3
import time
import serial
import sys
from datetime import datetime

def check_serial_port(com_port="COM5", baud=9600):
    """Test if serial port can be opened and basic communication works"""
    print(f"🔌 Testing serial port {com_port} at {baud} baud...")
    
    try:
        # Try to open the port
        ser = serial.Serial(com_port, baud, timeout=1.0)
        print(f"✅ Successfully opened {com_port}")
        
        # Test basic read/write
        test_data = b"TEST\n"
        ser.write(test_data)
        print(f"✅ Wrote test data: {test_data}")
        
        # Try to read (might timeout, which is OK)
        try:
            response = ser.read(100)
            if response:
                print(f"✅ Received response: {response}")
            else:
                print("⚠️  No response received (this might be normal)")
        except Exception as e:
            print(f"⚠️  Read timeout or error: {e}")
        
        ser.close()
        print(f"✅ Successfully closed {com_port}")
        return True
        
    except serial.SerialException as e:
        print(f"❌ Failed to open {com_port}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with {com_port}: {e}")
        return False

def check_database_access():
    """Test database creation and access"""
    print("\n🗄️  Testing database access...")
    
    db_paths = [
        "./sql/objective.db",
        "./sql/objective_memory.db", 
        "./sql/compositions.db"
    ]
    
    for db_path in db_paths:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Test connection
            conn = sqlite3.connect(db_path, timeout=10.0)
            cur = conn.cursor()
            
            # Test basic operations
            cur.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER, value TEXT)")
            cur.execute("INSERT INTO test VALUES (?, ?)", (1, "test_value"))
            cur.execute("SELECT * FROM test")
            result = cur.fetchone()
            cur.execute("DROP TABLE test")
            
            conn.commit()
            conn.close()
            
            print(f"✅ {db_path}: OK")
            
        except Exception as e:
            print(f"❌ {db_path}: {e}")

def check_file_permissions():
    """Check if we have proper file permissions"""
    print("\n📁 Testing file permissions...")
    
    test_dir = "./sql"
    test_file = "./sql/test_permissions.db"
    
    try:
        # Create directory
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ Created directory: {test_dir}")
        
        # Create test file
        with open(test_file, 'w') as f:
            f.write("test")
        print(f"✅ Created test file: {test_file}")
        
        # Read test file
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"✅ Read test file: {content}")
        
        # Delete test file
        os.remove(test_file)
        print(f"✅ Deleted test file: {test_file}")
        
    except Exception as e:
        print(f"❌ File permission error: {e}")

def check_process_communication():
    """Test basic inter-process communication"""
    print("\n🔄 Testing process communication...")
    
    try:
        import multiprocessing
        import queue
        
        # Test multiprocessing
        def test_worker(q):
            q.put("Hello from worker process")
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=test_worker, args=(q,))
        p.start()
        p.join(timeout=5)
        
        if p.is_alive():
            p.terminate()
            print("❌ Worker process didn't respond in time")
            return False
        
        try:
            result = q.get(timeout=1)
            print(f"✅ Process communication: {result}")
            return True
        except queue.Empty:
            print("❌ No message received from worker process")
            return False
            
    except Exception as e:
        print(f"❌ Process communication error: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n💻 Checking system resources...")
    
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"✅ CPU usage: {cpu_percent}%")
        
        # Memory usage
        memory = psutil.virtual_memory()
        print(f"✅ Memory usage: {memory.percent}% ({memory.available // (1024**3)} GB available)")
        
        # Disk space
        disk = psutil.disk_usage('.')
        print(f"✅ Disk usage: {disk.percent}% ({disk.free // (1024**3)} GB free)")
        
        return True
        
    except ImportError:
        print("⚠️  psutil not available, skipping system resource check")
        return True
    except Exception as e:
        print(f"❌ System resource check error: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 ZoMBI-Hop Communication Diagnostic Tool")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    tests = [
        ("Serial Port", lambda: check_serial_port()),
        ("Database Access", check_database_access),
        ("File Permissions", check_file_permissions),
        ("Process Communication", check_process_communication),
        ("System Resources", check_system_resources)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system should be ready for ZoMBI-Hop.")
    else:
        print("⚠️  Some tests failed. Please address the issues above before running ZoMBI-Hop.")
        
        # Provide specific recommendations
        print("\n🔧 RECOMMENDATIONS:")
        for test_name, result in results:
            if not result:
                if test_name == "Serial Port":
                    print("- Check if COM5 is the correct port")
                    print("- Ensure no other application is using the serial port")
                    print("- Try running as administrator if on Windows")
                elif test_name == "Database Access":
                    print("- Check if you have write permissions in the current directory")
                    print("- Ensure SQLite is properly installed")
                elif test_name == "File Permissions":
                    print("- Run the script as administrator")
                    print("- Check antivirus software isn't blocking file operations")
                elif test_name == "Process Communication":
                    print("- This might be a Windows-specific issue")
                    print("- Try running with different multiprocessing start method")

if __name__ == "__main__":
    main() 