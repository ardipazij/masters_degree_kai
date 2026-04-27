.include "m32def.inc"
.cseg
.org 0

main:

	ldi r16, 0x00
	out DDRA, r16      ; Ввод через порт А
	ldi r16, 0xFF
	out DDRD, r16      ; Вывод через порт D

loop:
	
	in r17, PINA        ; Считывание байта с порта А

	mov r18, r17   
	com r18				; инверсия
	andi r18, 0x01		;r0
	mov r19, r17
	lsr r19
	andi r19, 0x01		;r1
	mov r20, r17
	lsr r20
	lsr r20
	andi r20, 0x01		;r2
	
	and r18, r19		;не(r0) & r1
	or  r18, r20		;не(r0) & r1 V r2

	out PORTD, r18      ; Вывод результата в порт D

	rjmp loop           ; Бесконечный цикл
