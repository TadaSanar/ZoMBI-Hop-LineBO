import serial
import sqlite3
import time
import numpy as np
import json
import threading


def reset_objective(db_path="./sql/objective.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find and drop the first table if it exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbl = cursor.fetchone()

    if tbl:
        table_name = tbl[0]
        cursor.execute(f"DELETE FROM {table_name};")
        conn.commit()

    conn.close()


def write_compositions(start,
                       end,
                       array,
                       timestamp,
                       start_cache,
                       end_cache,
                       array_cache,
                       db_path="./sql/compositions.db"):
    """
    Overwrite the following tables in compositions.db:
      - compositions       : the main array (n×10)
      - start              : the main 1×10 start vector
      - end                : the main 1×10 end   vector
      - compositions_cache : the cached array (n×10)
      - start_cache        : the cached 1×10 start vector
      - end_cache          : the cached 1×10 end   vector
      - timestamp          : a single REAL ts
    """
    # -- shape validations --
    for name, arr in [("array", array), ("array_cache", array_cache)]:
        if arr.ndim != 2 or arr.shape[1] != 10:
            raise ValueError(f"`{name}` must be 2D with exactly 10 columns")

    for name, vec in [
        ("start", start), ("end", end),
        ("start_cache", start_cache), ("end_cache", end_cache)
    ]:
        if vec.ndim != 1 or vec.shape[0] != 10:
            raise ValueError(f"`{name}` must be 1D with length 10")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # reusable column defs and placeholders
    col_defs    = ", ".join(f"col_{i} REAL" for i in range(10))
    placeholders = ", ".join("?" for _ in range(10))

    # 1) main compositions
    cur.execute("DROP TABLE IF EXISTS compositions")
    cur.execute(f"CREATE TABLE compositions ({col_defs})")
    cur.executemany(
        f"INSERT INTO compositions VALUES ({placeholders})",
        array.tolist()
    )

    # 2) main start
    cur.execute("DROP TABLE IF EXISTS start")
    cur.execute(f"CREATE TABLE start ({col_defs})")
    cur.execute(
        f"INSERT INTO start VALUES ({placeholders})",
        start.tolist()
    )

    # 3) main end
    cur.execute("DROP TABLE IF EXISTS end")
    cur.execute(f"CREATE TABLE end ({col_defs})")
    cur.execute(
        f"INSERT INTO end VALUES ({placeholders})",
        end.tolist()
    )

    # 4) cache compositions
    cur.execute("DROP TABLE IF EXISTS compositions_cache")
    cur.execute(f"CREATE TABLE compositions_cache ({col_defs})")
    cur.executemany(
        f"INSERT INTO compositions_cache VALUES ({placeholders})",
        array_cache.tolist()
    )

    # 5) cache start
    cur.execute("DROP TABLE IF EXISTS start_cache")
    cur.execute(f"CREATE TABLE start_cache ({col_defs})")
    cur.execute(
        f"INSERT INTO start_cache VALUES ({placeholders})",
        start_cache.tolist()
    )

    # 6) cache end
    cur.execute("DROP TABLE IF EXISTS end_cache")
    cur.execute(f"CREATE TABLE end_cache ({col_defs})")
    cur.execute(
        f"INSERT INTO end_cache VALUES ({placeholders})",
        end_cache.tolist()
    )

    # 7) timestamp
    cur.execute("DROP TABLE IF EXISTS timestamp")
    cur.execute("CREATE TABLE timestamp (ts REAL)")
    cur.execute("INSERT INTO timestamp VALUES (?)", (float(timestamp),))

    conn.commit()
    conn.close()
    

# # — 3) reset_compositions drops all four tables —
def reset_compositions(db_path="./sql/compositions.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for tbl in ("compositions", "start", "end", "timestamp"):
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()


# ─── Helper to drop & recreate the 'results' table with one row ─────────────
def add_vector_to_first_row(x, db_path):
    x = np.asarray(x).ravel()
    N = x.shape[0]
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS objective")
    cols = ", ".join(f"v{i} REAL" for i in range(N))
    cur.execute(f"CREATE TABLE objective ({cols})")
    ph   = ", ".join("?" for _ in range(N))
    cur.execute(f"INSERT INTO objective VALUES ({ph})", tuple(x.tolist()))
    conn.commit()
    conn.close()


# # — THREAD 2: read compositions.db and emit JSON packets —
# def composition_sender(ser, hz, comp_db_path):
#     pause = 1.0/hz

#     def read_all():
#         conn = sqlite3.connect(comp_db_path); cur = conn.cursor()
#         cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp'")
#         if not cur.fetchone():
#             conn.close(); return None
#         cur.execute("SELECT ts FROM timestamp LIMIT 1"); row = cur.fetchone()
#         if not row:
#             conn.close(); return None
#         ts = row[0]
#         def tbl(n): 
#             cur.execute(f"SELECT * FROM {n}"); return cur.fetchall()
#         start = tbl("start"); end = tbl("end"); comps = tbl("compositions")
#         conn.close()
#         if not (start and end and comps):
#             return None
#         return ts, start[0], end[0], comps

#     while True:
#         data = read_all()
#         if data:
#             ts, start, end, comps = data
#             pkt = {
#               "type":"composition",
#               "ts":ts,
#               "start": list(start),
#               "end":   list(end),
#               "comps": [list(r) for r in comps]
#             }
#             ser.write((json.dumps(pkt)+"\n").encode())
#             print(f"[composition_sender] sent ts={ts}")
#         time.sleep(pause)


# ─── THREAD: read compositions.db (including caches) and emit JSON packets ──
def composition_sender(ser, hz, comp_db_path, chaos=False, verbose=True):
    """
    Every 1/hz seconds, polls all six tables in compositions.db and 
    once all are non‐empty, emits a single JSON packet with:
      { type:"composition", ts, start, end, comps, start_cache, end_cache, compositions_cache }
    """
    ser.timeout = 1.0 / hz
    pause = 1.0 / hz
    next_send = time.monotonic()

    def read_all():
        try:
            conn = sqlite3.connect(comp_db_path)
            cur  = conn.cursor()
            # timestamp
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp'")
            if not cur.fetchone():
                return None
            cur.execute("SELECT ts FROM timestamp LIMIT 1")
            trow = cur.fetchone()
            if not trow:
                return None
            ts = trow[0]

            # helper to fetch
            def fetch(tbl):
                cur.execute(f"SELECT * FROM {tbl}")
                return cur.fetchall()

            start       = fetch("start")
            end         = fetch("end")
            comps       = fetch("compositions")
            sc          = fetch("start_cache")
            ec          = fetch("end_cache")
            cc          = fetch("compositions_cache")
            conn.close()

            if not (start and end and comps and sc and ec and cc):
                return None

            return ts, start[0], end[0], comps, sc[0], ec[0], cc
        except Exception:
            return None

    while True:
        # maintain exact hz
        now = time.monotonic()
        if now < next_send:
            time.sleep(next_send - now)
        next_send += pause

        data = read_all()
        if not data:
            continue

        ts, start, end, comps, sc, ec, cc = data
        if chaos:
            ts = np.random.randint(1,10)
        pkt = {
            "type":              "composition",
            "ts":                ts,
            "start":             list(start),
            "end":               list(end),
            "comps":             [list(r) for r in comps],
            "start_cache":       list(sc),
            "end_cache":         list(ec),
            "compositions_cache":[list(r) for r in cc]
        }

        raw = (json.dumps(pkt) + "\n").encode()
        try:
            ser.write(raw)
            if verbose:
                print(f"[composition_sender] sent ts={ts}")
        except serial.SerialException as e:
            if verbose:
                print(f"[composition_sender] SerialException on write: {e!r}, resetting output buffer.")
            try: ser.reset_output_buffer()
            except: pass
            # will retry next cycle


# ─── Helper to write a 1D objective row ─────────────────────────────────────
def _write_objective_row(vals, db_path):
    N = len(vals)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS objective")
    cols = ", ".join(f"v{i} REAL" for i in range(N))
    cur.execute(f"CREATE TABLE objective ({cols})")
    ph = ", ".join("?" for _ in range(N))
    cur.execute(f"INSERT INTO objective VALUES ({ph})", tuple(vals))
    conn.commit()
    conn.close()

# ─── Helper to write an n×M compositions matrix ────────────────────────────
def _write_compositions_matrix(mat, db_path):
    mat = np.asarray(mat, dtype=float)
    n, M = mat.shape
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS compositions")
    cols = ", ".join(f"c{i} REAL" for i in range(M))
    cur.execute(f"CREATE TABLE compositions ({cols})")
    ph = ", ".join("?" for _ in range(M))
    cur.executemany(f"INSERT INTO compositions VALUES ({ph})", mat.tolist())
    conn.commit()
    conn.close()


# ─── THREAD: receive JSON objective, write to objective.db ────────────────
def objective_receiver(ser, hz, obj_db_path, mem_db_path, verbose=True):
    """
    Listens for {"type":"objective","values":[…],"comps":[[…],…]} lines.
    Writes new objectives into obj_db_path & mem_db_path (only if memory-diff & obj-db empty),
    and (if provided) their compositions into obj_db_path.
    """
    ser.timeout = 1.0 / hz
    pause = 1.0 / hz
    next_recv = time.monotonic()

    while True:
        # maintain exact hz
        now = time.monotonic()
        if now < next_recv:
            time.sleep(next_recv - now)
        next_recv += pause

        # 1) read raw line
        try:
            raw_bytes = ser.readline()
        except serial.SerialException as e:
            if verbose:
                print(f"[objective_receiver] SerialException: {e!r}, resetting input buffer.")
            try: ser.reset_input_buffer()
            except: pass
            continue

        if not raw_bytes:
            continue

        try:
            raw = raw_bytes.decode("utf-8", errors="ignore").strip()
            pkt = json.loads(raw)
        except Exception:
            # either decode or JSON failed
            continue

        if pkt.get("type") != "objective" or "values" not in pkt:
            continue

        vals  = np.asarray(pkt["values"], dtype=float)
        comps = pkt.get("comps", None)

        # — memory diff check — 
        try:
            conn = sqlite3.connect(mem_db_path); c = conn.cursor()
            c.execute("SELECT * FROM objective")
            mem = c.fetchone()
            conn.close()
            memory_diff = not (mem and np.allclose(mem, vals))
        except Exception as e:
            if verbose:
                print(f"[objective_receiver] ERROR checking memory DB: {e!r}")
            continue

        # — objective empty check —
        try:
            conn = sqlite3.connect(obj_db_path); c = conn.cursor()
            c.execute("SELECT * FROM objective")
            obj = c.fetchone()
            conn.close()
            obj_empty = (obj is None)
        except Exception as e:
            if verbose:
                print(f"[objective_receiver] ERROR checking objective DB: {e!r}")
            continue

        if memory_diff and obj_empty:
            # write both new objective & memory
            try:
                _write_objective_row(vals.tolist(), obj_db_path)
                _write_objective_row(vals.tolist(), mem_db_path)
                if verbose:
                    print("[objective_receiver] Wrote new objective:", vals)
            except Exception as e:
                if verbose:
                    print(f"[objective_receiver] ERROR writing objective: {e!r}")

            # optionally write compositions
            if isinstance(comps, list) and comps and isinstance(comps[0], list):
                try:
                    _write_compositions_matrix(comps, obj_db_path)
                    if verbose:
                        print(f"[objective_receiver] Wrote compositions ({len(comps)}×{len(comps[0])})")
                except Exception as e:
                    if verbose:
                        print(f"[objective_receiver] ERROR writing compositions: {e!r}")
        else:
            if verbose:
                print(f"[objective_receiver] Skip → diff={memory_diff}, empty={obj_empty}")



# ─── Orchestrator: single Serial + two threads ─────────────────────────────
def start_serial_dual_io_shared_port(COM, baud,
                                     obj_hz=5.0,
                                     comp_hz=1.0,
                                     obj_db ="./sql/objective.db",
                                     mem_db ="./sql/objective_memory.db",
                                     comp_db="./sql/compositions.db",
                                     chaos=True):
    ser = serial.Serial(COM, baud, timeout=1)
    print(f"[Machine2] Opened {COM}@{baud}")

    # Receiver: objective → write to both obj_db and mem_db
    t_o = threading.Thread(
        target=objective_receiver,
        args=(ser, obj_hz, obj_db, mem_db),
        daemon=True
    )
    # Sender: read compositions + caches → emit JSON
    t_c = threading.Thread(
        target=composition_sender,
        args=(ser, comp_hz, comp_db, chaos),
        daemon=True
    )
    t_o.start()
    t_c.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Machine2] Closing…")
        ser.close()