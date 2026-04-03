; lazerdillo dac and timer
; fe40 fe41 mode2 5120hz
; a=1 b=3 base=26.64 f_x=26.64 f_y=79.92 stepx=341 stepy=1023 phase=0.0

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
    mov phaseY_lo, #00h
    mov phaseY_hi, #00h
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
    add a, #55h
    mov phaseX_lo, a
    mov a, phaseX_hi
    addc a, #01h
    mov phaseX_hi, a
    ; add step to phase y
    mov a, phaseY_lo
    add a, #0FFh
    mov phaseY_lo, a
    mov a, phaseY_hi
    addc a, #03h
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
    .db 80h, 81h, 82h, 83h, 84h, 85h, 86h, 87h, 88h, 89h, 8Ah, 8Bh, 8Ch, 8Dh, 8Dh, 8Eh
    .db 8Fh, 90h, 91h, 92h, 93h, 94h, 95h, 95h, 96h, 97h, 98h, 99h, 99h, 9Ah, 9Bh, 9Ch
    .db 9Ch, 9Dh, 9Eh, 9Eh, 9Fh, 0A0h, 0A0h, 0A1h, 0A1h, 0A2h, 0A2h, 0A3h, 0A3h, 0A4h, 0A4h, 0A5h
    .db 0A5h, 0A5h, 0A6h, 0A6h, 0A6h, 0A7h, 0A7h, 0A7h, 0A7h, 0A7h, 0A8h, 0A8h, 0A8h, 0A8h, 0A8h, 0A8h
    .db 0A8h, 0A8h, 0A8h, 0A8h, 0A8h, 0A8h, 0A8h, 0A7h, 0A7h, 0A7h, 0A7h, 0A7h, 0A6h, 0A6h, 0A6h, 0A5h
    .db 0A5h, 0A5h, 0A4h, 0A4h, 0A3h, 0A3h, 0A2h, 0A2h, 0A1h, 0A1h, 0A0h, 0A0h, 9Fh, 9Eh, 9Eh, 9Dh
    .db 9Ch, 9Ch, 9Bh, 9Ah, 99h, 99h, 98h, 97h, 96h, 95h, 95h, 94h, 93h, 92h, 91h, 90h
    .db 8Fh, 8Eh, 8Dh, 8Dh, 8Ch, 8Bh, 8Ah, 89h, 88h, 87h, 86h, 85h, 84h, 83h, 82h, 81h
    .db 80h, 7Fh, 7Eh, 7Dh, 7Ch, 7Bh, 7Ah, 79h, 78h, 77h, 76h, 75h, 74h, 73h, 73h, 72h
    .db 71h, 70h, 6Fh, 6Eh, 6Dh, 6Ch, 6Bh, 6Bh, 6Ah, 69h, 68h, 67h, 67h, 66h, 65h, 64h
    .db 64h, 63h, 62h, 62h, 61h, 60h, 60h, 5Fh, 5Fh, 5Eh, 5Eh, 5Dh, 5Dh, 5Ch, 5Ch, 5Bh
    .db 5Bh, 5Bh, 5Ah, 5Ah, 5Ah, 59h, 59h, 59h, 59h, 59h, 58h, 58h, 58h, 58h, 58h, 58h
    .db 58h, 58h, 58h, 58h, 58h, 58h, 58h, 59h, 59h, 59h, 59h, 59h, 5Ah, 5Ah, 5Ah, 5Bh
    .db 5Bh, 5Bh, 5Ch, 5Ch, 5Dh, 5Dh, 5Eh, 5Eh, 5Fh, 5Fh, 60h, 60h, 61h, 62h, 62h, 63h
    .db 64h, 64h, 65h, 66h, 67h, 67h, 68h, 69h, 6Ah, 6Bh, 6Bh, 6Ch, 6Dh, 6Eh, 6Fh, 70h
    .db 71h, 72h, 73h, 73h, 74h, 75h, 76h, 77h, 78h, 79h, 7Ah, 7Bh, 7Ch, 7Dh, 7Eh, 7Fh
