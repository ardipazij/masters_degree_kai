.include "m32def.inc" // подключение определений ATmega32
.cseg
.org 0
			cbi		DDRA,	DDA0	//установка нуля - вход
			cbi		DDRA,	DDA1	//вход
			sbi		DDRA,	DDA2	//выход (1)
			cbi 	DDRA,	DDA3	//вход
			sbi 	PORTA,	PA0		//кнопка 0 опущена
			sbi		PORTA,	PA1		//кнопка 1 опущена
			cbi		PORTA,	PA2		//светодиод 2 выключен
			cbi 	PORTA,	PA3		//?!?!? (в идеале выше его бы в 1 поставить)
loop:		sbis	PINA, PINA0		//Прожата ли кнопка 0? (не нажата - пропуск)
			rjmp	end				//Переход к end
			sbis	PINA, PINA1		//Прожата ли кнопка 1? (не нажата - пропуск)
			rjmp	setPA2			//Переход
									// Сюда попадаем если ОБЕ кнопки отпущены
setPA3:		cbi		PORTA,	PA2		//Выключить светодиод 2
			sbi		PORTA,	PA3		//Включить кнопку 3
			rjmp	loop			//Перейти на loop
setPA2:		sbi		PORTA,	PA2		//Включить светодиод 2
			cbi		PORTA,	PA3		//Выключить кнопку 3
			rjmp	loop			//Перейти на loop
end:		cbi		PORTA,	PA2		//Выключить светодиод 2
			cbi		PORTA,	PA3		//Выключить кнопку 3
