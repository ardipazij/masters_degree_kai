.include "m32def.inc"    ; ??????????? ??????????? ??? ATmega32 (???????, ?????, ????)
.cseg                     ; 
.org 0                     ; 

main:
        ldi r16, 0x00    ; инициализация
        out DDRC, r16    ;

        ldi r16, 0xFF    ; 
        out DDRD, r16    ; 

loop:
        in r17, PINC     ; считывание байта с порта С в R17

        mov r18, r17     ; копируем
        andi r18, 0x03   ; по маске оставляем только 2 младших бита

        mov r19, r17     ; копирование
        lsr r19          ; 
        lsr r19          ; сдвиги
        swap r19         ;
        andi r19, 0x03   ; 

        sub r18, r19     ; 
        out PORTD, r18   ; 

        rjmp loop        ;
