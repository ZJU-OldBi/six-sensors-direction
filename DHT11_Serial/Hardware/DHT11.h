#ifndef __DHT11_H
#define __DHT11_H

#include "stm32f10x.h"

/* DHT11 数据引脚：默认 PB5，可按需修改下面三行 */
#define DHT11_GPIO_PORT		GPIOB
#define DHT11_GPIO_PIN		GPIO_Pin_5
#define DHT11_GPIO_CLK		RCC_APB2Periph_GPIOB

void DHT11_Init(void);
uint8_t DHT11_ReadData(uint8_t *Temperature, uint8_t *Humidity);

#endif
