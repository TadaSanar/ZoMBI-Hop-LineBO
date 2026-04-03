#!/usr/bin/env python
# lissajous asm for lazerdillo
# for circle: python lissajous_gen.py 1 1 --amplitude 40 --phase 0.25 -o lazer_circle.asm
# for tie: python lissajous_gen.py 1 3 --amplitude 40 --phase 0.25 -o lazer_tie.asm
# for other shape: python lissajous_gen.py 1 4 --amplitude 40 -o lazer_a1b4phi0.asm

import argparse
import math
import sys

INTERRUPT_RATE = 5120
PHASE_FULL = 65536
MAX_FREQ_HZ = 80
MIN_FIGURE_RATE_HZ = 20  # min figure repeat hz
BYTES_PER_LINE = 16
N = 256


def hex_byte(b: int) -> str:
    # 8051 style hex byte
    out = f"{b:02X}h"
    return "0" + out if out[0] in "ABCDEF" else out


def build_sine_table(amplitude: float = 127.5) -> list[int]:
    # one period centered at 128
    amp = max(0, min(127.5, amplitude))
    out = []
    for i in range(N):
        val = 128 + amp * math.sin(2 * math.pi * i / N)
        out.append(max(0, min(255, round(val))))
    return out


def sine_table_db_lines(samples: list[int]) -> list[str]:
    # sixteen bytes per .db line
    lines = []
    for start in range(0, N, BYTES_PER_LINE):
        chunk = samples[start : start + BYTES_PER_LINE]
        hex_str = ", ".join(hex_byte(b) for b in chunk)
        lines.append(f"    .db {hex_str}")
    return lines


def compute_step(freq_hz: float) -> int:
    # phase increment per tick
    step = round(PHASE_FULL * freq_hz / INTERRUPT_RATE)
    return max(0, min(PHASE_FULL-1, step))


def compute_steps_exact_ratio(a: int, b: int) -> tuple[int, int]:
    # exact ratio largest k steps
    max_step = int(PHASE_FULL * MAX_FREQ_HZ / INTERRUPT_RATE)
    k = min(max_step // a, max_step // b)
    k = max(1, k)
    return k * a, k * b


def validate_lissajous(
    a: int,
    b: int,
    f_x: float,
    f_y: float,
    step_x: int,
) -> float:
    # mirror frequency ceiling
    if max(f_x, f_y) > MAX_FREQ_HZ + 0.01:
        raise ValueError(
            f"Frequency too high for mirrors: f_x={f_x:.2f} Hz, f_y={f_y:.2f} Hz "
            f"(max {MAX_FREQ_HZ} Hz). Use smaller a/b."
        )
    g = math.gcd(a, b)
    a_r = a // g
    b_r = b // g
    # closure rate: f0 = f_x/a' = f_y/b' with reduced ratio a':b'
    f0_from_x = f_x / a_r
    f0_from_y = f_y / b_r
    if abs(f0_from_x - f0_from_y) > 0.02:
        raise ValueError(
            f"Figure rate from X ({f0_from_x:.4f} Hz) vs Y ({f0_from_y:.4f} Hz) disagree."
        )
    f0 = (f0_from_x + f0_from_y) / 2
    k_step = step_x // a
    f0_from_steps = k_step * g * INTERRUPT_RATE / PHASE_FULL
    if abs(f0 - f0_from_steps) > 0.02:
        raise ValueError(
            f"Figure rate from freqs ({f0:.4f} Hz) vs steps ({f0_from_steps:.4f} Hz) disagree."
        )
    # figure repeat rate floor
    if f0 < MIN_FIGURE_RATE_HZ - 0.01:
        raise ValueError(
            f"Figure period too long: repetition rate {f0:.2f} Hz "
            f"(need >= {MIN_FIGURE_RATE_HZ} Hz). Use a simpler a:b ratio."
        )
    return f0


def generate_asm(
    a: int,
    b: int,
    phase: float = 0.0,
    amplitude: float = 96.0,
) -> tuple[str, dict]:
    # exact ratio step pair
    step_x, step_y = compute_steps_exact_ratio(a, b)
    f_x_actual = step_x * INTERRUPT_RATE / PHASE_FULL
    f_y_actual = step_y * INTERRUPT_RATE / PHASE_FULL
    figure_rate_hz = validate_lissajous(a, b, f_x_actual, f_y_actual, step_x)
    phase_y0 = round(PHASE_FULL * phase) & 0xFFFF

    step_x_lo = step_x & 0xFF
    step_x_hi = (step_x >> 8) & 0xFF
    step_y_lo = step_y & 0xFF
    step_y_hi = (step_y >> 8) & 0xFF
    phase_y0_lo = phase_y0 & 0xFF
    phase_y0_hi = (phase_y0 >> 8) & 0xFF

    samples = build_sine_table(amplitude)
    sine_table = "\n".join(sine_table_db_lines(samples))

    base_actual = f_x_actual / a
    comment = f"; a={a} b={b} base={base_actual:.2f} f_x={f_x_actual:.2f} f_y={f_y_actual:.2f} stepx={step_x} stepy={step_y} phase={phase}"

    asm_template = """
; lazerdillo dac and timer
; fe40 fe41 mode2 5120hz
{comment}

; reset and irq vectors
    .org 0
    ljmp main
    ; t0 vector later below

; symbol addresses and temps
.equ DAC_X, 0FE40h
.equ DAC_Y, 0FE41h
.equ TH0_RELOAD, 04Ch
.equ phaseX_lo, 30h
.equ phaseX_hi, 31h
.equ phaseY_lo, 32h
.equ phaseY_hi, 33h
.equ t0_tick, 34h

; entry at 1000h run
; mode2 eight bit autoreload
    .org 1000h
main:
    mov phaseX_lo, #0
    mov phaseX_hi, #0
    mov phaseY_lo, #{phase_y0_lo}
    mov phaseY_hi, #{phase_y0_hi}
    mov t0_tick, #0
    mov tmod, #02h
    mov th0, #TH0_RELOAD
    mov tl0, #TH0_RELOAD
    setb et0
    setb ea
    setb tr0
loop:
    mov a, t0_tick
    jz loop
    mov t0_tick, #0
    ; add step to phase x
    mov a, phaseX_lo
    add a, #{step_x_lo}
    mov phaseX_lo, a
    mov a, phaseX_hi
    addc a, #{step_x_hi}
    mov phaseX_hi, a
    ; add step to phase y
    mov a, phaseY_lo
    add a, #{step_y_lo}
    mov phaseY_lo, a
    mov a, phaseY_hi
    addc a, #{step_y_hi}
    mov phaseY_hi, a
    ; sine both dacs
    mov dptr, #sine_tbl
    mov a, phaseX_hi
    movc a, @a+dptr
    mov dptr, #DAC_X
    movx @dptr, a
    mov dptr, #sine_tbl
    mov a, phaseY_hi
    movc a, @a+dptr
    mov dptr, #DAC_Y
    movx @dptr, a
    sjmp loop

; t0 sets tick byte
    .org 0Bh
    mov t0_tick, #1
    reti

; rom sine 256 samples
sine_tbl:
{sine_table}
"""
    asm = asm_template.format(
        comment=comment,
        phase_y0_lo=hex_byte(phase_y0_lo),
        phase_y0_hi=hex_byte(phase_y0_hi),
        step_x_lo=hex_byte(step_x_lo),
        step_x_hi=hex_byte(step_x_hi),
        step_y_lo=hex_byte(step_y_lo),
        step_y_hi=hex_byte(step_y_hi),
        sine_table=sine_table,
    ).strip() + "\n"
    info = {
        "f_x_actual": f_x_actual,
        "f_y_actual": f_y_actual,
        "figure_rate_hz": figure_rate_hz,
    }
    return asm, info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="emit lissajous r31jp asm",
    )
    parser.add_argument("a", type=int, help="x ratio numerator")
    parser.add_argument("b", type=int, help="y ratio denominator")
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        metavar="PHI",
        help="y phase 0-1 * pi radians",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=96.0,
        metavar="A",
        help="sime amplitude 0-127.5",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="write asm file path",
    )
    args = parser.parse_args()
    try:
        asm, info = generate_asm(
            a=args.a,
            b=args.b,
            phase=args.phase,
            amplitude=args.amplitude,
        )
    except ValueError as e:
        print(f"lissajous_gen: {e}", file=sys.stderr)
        return 1
    print(
        f"actual: f_x={info['f_x_actual']:.2f} Hz, f_y={info['f_y_actual']:.2f} Hz, figure={info['figure_rate_hz']:.2f} Hz",
        file=sys.stderr,
    )
    out_path = args.output
    if out_path:
        with open(out_path, "w", newline="\n") as f:
            f.write(asm)
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        print(asm)
        print(end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
