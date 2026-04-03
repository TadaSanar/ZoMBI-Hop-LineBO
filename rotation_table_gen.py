#!/usr/bin/env python3
# rotation_table lines for lazer_var (speed 0-9)

from lissajous_gen import compute_step, hex_byte

MAX_ROT_HZ = 1.5
N_SPEED = 10


def main() -> None:
    print("; rotation step pairs speed zero nine")
    print("rotation_table:")
    denom = N_SPEED - 1
    for k in range(N_SPEED):
        f_hz = MAX_ROT_HZ * k / denom
        step = compute_step(f_hz)
        lo = hex_byte(step & 0xFF)
        hi = hex_byte((step >> 8) & 0xFF)
        print(f"    .db {lo}, {hi}  ; k={k} {f_hz:.3f}hz step={step}")


if __name__ == "__main__":
    main()
