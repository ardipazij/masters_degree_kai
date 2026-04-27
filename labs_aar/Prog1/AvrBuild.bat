@ECHO OFF
"D:\AvrAssembler2\avrasm2.exe" -S "D:\Progs_AVR\Prog1\labels.tmp" -fI -W+ie -C V2E -o "D:\Progs_AVR\Prog1\Prog1.hex" -d "D:\Progs_AVR\Prog1\Prog1.obj" -e "D:\Progs_AVR\Prog1\Prog1.eep" -m "D:\Progs_AVR\Prog1\Prog1.map" "D:\Progs_AVR\Prog1\Prog1.asm"
