#include "stm32f10x.h"
#include "Delay.h"
#include "Serial.h"
#include "DHT11.h"

int main(void)
{
	uint8_t Temperature;
	uint8_t Humidity;

	Serial_Init();
	DHT11_Init();

	Serial_Printf("\r\n");
	Serial_Printf("DHT11 serial demo\r\n");
	Serial_Printf("USART1: PA9(TX), 9600 8N1\r\n");
	Serial_Printf("DHT11 DATA: PB5\r\n\r\n");

	/* DHT11 needs about 1 s to become stable after power-on */
	Delay_ms(1000);

	while (1)
	{
		if (DHT11_ReadData(&Temperature, &Humidity))
		{
			Serial_Printf("Temperature: %d C, Humidity: %d %%RH\r\n", Temperature, Humidity);
		}
		else
		{
			Serial_Printf("DHT11 read error\r\n");
		}

		/* DHT11 sampling interval must be at least 1 s */
		Delay_ms(2000);
	}
}
