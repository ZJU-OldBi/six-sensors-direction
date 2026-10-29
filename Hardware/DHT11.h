#ifndef __DHT11_H
#define __DHT11_H

#include "stm32f10x.h"

/* DHT11 数据引脚：默认 PB5，可按需修改下面三行 */
#define DHT11_GPIO_PORT		GPIOB
#define DHT11_GPIO_PIN		GPIO_Pin_5
#define DHT11_GPIO_CLK		RCC_APB2Periph_GPIOB

/* 最近一次读取的错误码，便于串口诊断：
 * 0=成功，1=没有等到响应低电平，2=没有等到响应高电平，
 * 3=响应高电平没有结束，4=数据位等待超时，5=校验和错误，
 * 6=总线一直被拉低（检查接线/上拉/传感器是否损坏）
 */
extern volatile uint8_t DHT11_ErrorCode;

/* 最近一次读取到的原始 5 字节，校验失败时可用来判断时序是否基本正确 */
extern volatile uint8_t DHT11_RawData[5];

void DHT11_Init(void);
uint8_t DHT11_ReadData(uint8_t *Temperature, uint8_t *Humidity);

#endif
