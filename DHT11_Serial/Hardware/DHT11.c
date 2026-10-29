#include "stm32f10x.h"
#include "DHT11.h"
#include "Delay.h"

/* 等待总线电平变化的超时时间（单位：次，1 次约 1us） */
#define DHT11_TIMEOUT	100

/* 将数据引脚配置为推挽输出 */
static void DHT11_PinOutput(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Pin = DHT11_GPIO_PIN;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(DHT11_GPIO_PORT, &GPIO_InitStructure);
}

/* 将数据引脚配置为上拉输入（释放总线，由上拉电阻保持高电平） */
static void DHT11_PinInput(void)
{
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStructure.GPIO_Pin = DHT11_GPIO_PIN;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(DHT11_GPIO_PORT, &GPIO_InitStructure);
}

/* 等待数据引脚变为指定电平，超时返回 0，成功返回 1 */
static uint8_t DHT11_WaitLevel(BitAction Level, uint16_t Timeout)
{
	while (GPIO_ReadInputDataBit(DHT11_GPIO_PORT, DHT11_GPIO_PIN) != Level)
	{
		if (Timeout-- == 0)
		{
			return 0;
		}
		Delay_us(1);
	}
	return 1;
}

/**
  * @brief  初始化 DHT11 数据引脚
  * @param  无
  * @retval 无
  */
void DHT11_Init(void)
{
	RCC_APB2PeriphClockCmd(DHT11_GPIO_CLK, ENABLE);
	
	DHT11_PinOutput();
	GPIO_SetBits(DHT11_GPIO_PORT, DHT11_GPIO_PIN);	// 空闲时保持高电平
}

/**
  * @brief  读取 DHT11 的湿度和温度
  * @param  Temperature 温度（℃），整数部分
  * @param  Humidity 湿度（%RH），整数部分
  * @retval 1：读取成功；0：无响应或校验失败
  * @note   DHT11 两次读取间隔应不小于 1 秒
  */
uint8_t DHT11_ReadData(uint8_t *Temperature, uint8_t *Humidity)
{
	uint8_t Data[5] = {0};
	uint8_t i, j;
	uint32_t Primask;
	
	/* 主机发送起始信号：拉低总线至少 18ms */
	DHT11_PinOutput();
	GPIO_ResetBits(DHT11_GPIO_PORT, DHT11_GPIO_PIN);
	Delay_ms(20);
	
	/* 释放总线并切换为输入，等待 DHT11 响应 */
	GPIO_SetBits(DHT11_GPIO_PORT, DHT11_GPIO_PIN);
	DHT11_PinInput();
	Delay_us(30);
	
	/* 读取时序较敏感，临时关闭中断，避免中断拉长采样间隔 */
	Primask = __get_PRIMASK();
	__disable_irq();
	
	/* DHT11 响应：低电平 80us + 高电平 80us */
	if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0) goto dht11_error;
	if (DHT11_WaitLevel(Bit_SET, DHT11_TIMEOUT) == 0) goto dht11_error;
	if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0) goto dht11_error;
	
	/* 连续读取 40 位数据：5 字节 */
	for (i = 0; i < 5; i++)
	{
		for (j = 0; j < 8; j++)
		{
			/* 每位以 50us 低电平开始，等待低电平结束 */
			if (DHT11_WaitLevel(Bit_SET, DHT11_TIMEOUT) == 0) goto dht11_error;
			
			/* 高电平持续 26~28us 表示 0，持续 70us 表示 1 */
			Delay_us(40);
			Data[i] <<= 1;
			if (GPIO_ReadInputDataBit(DHT11_GPIO_PORT, DHT11_GPIO_PIN) == Bit_SET)
			{
				Data[i] |= 0x01;
				/* 等待该位的高电平结束 */
				if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0) goto dht11_error;
			}
		}
	}
	
	if (!Primask)
	{
		__enable_irq();
	}
	
	/* 校验：前 4 字节之和的低 8 位等于第 5 字节 */
	if ((uint8_t)(Data[0] + Data[1] + Data[2] + Data[3]) != Data[4])
	{
		return 0;
	}
	
	*Humidity = Data[0];		// 湿度整数部分
	*Temperature = Data[2];		// 温度整数部分
	return 1;
	
dht11_error:
	if (!Primask)
	{
		__enable_irq();
	}
	return 0;
}
