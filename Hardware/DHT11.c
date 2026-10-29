#include "stm32f10x.h"
#include "DHT11.h"
#include "Delay.h"

/* 等待总线电平变化的超时时间（单位：次，1 次约 1us） */
#define DHT11_TIMEOUT	200

/* 诊断信息：最近一次读取的错误码和原始数据 */
volatile uint8_t DHT11_ErrorCode = 0;
volatile uint8_t DHT11_RawData[5] = {0};

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
	
	/* 清除上一次的诊断信息 */
	DHT11_ErrorCode = 0;
	for (i = 0; i < 5; i++)
	{
		DHT11_RawData[i] = 0;
	}
	
	/* 先释放总线并检查空闲电平：正常应为高电平。
	 * 如果一直为低，说明数据线短路、传感器没供电或传感器损坏。 */
	DHT11_PinInput();
	if (GPIO_ReadInputDataBit(DHT11_GPIO_PORT, DHT11_GPIO_PIN) == Bit_RESET)
	{
		DHT11_ErrorCode = 6;
		return 0;
	}
	
	/* 主机发送起始信号：拉低总线至少 18ms */
	DHT11_PinOutput();
	GPIO_ResetBits(DHT11_GPIO_PORT, DHT11_GPIO_PIN);
	Delay_ms(20);
	
	/* 从释放总线之前就关闭中断，避免串口等中断把 30us 延时拉长、
	 * 从而错过 DHT11 的 80us 响应低电平。起始低电平的 20ms 不受影响。 */
	Primask = __get_PRIMASK();
	__disable_irq();
	
	/* 释放总线并切换为输入。这里不再固定延时 30us，而是直接等待响应低电平；
	 * 即使 Delay_us 因时钟或中断略有偏差，也不会错过 80us 的响应低电平。 */
	GPIO_SetBits(DHT11_GPIO_PORT, DHT11_GPIO_PIN);
	DHT11_PinInput();
	
	/* DHT11 响应：低电平 80us + 高电平 80us */
	if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0)
	{
		DHT11_ErrorCode = 1;		/* 没有等到响应低电平：传感器没响应 */
		goto dht11_error;
	}
	if (DHT11_WaitLevel(Bit_SET, DHT11_TIMEOUT) == 0)
	{
		DHT11_ErrorCode = 2;
		goto dht11_error;
	}
	if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0)
	{
		DHT11_ErrorCode = 3;
		goto dht11_error;
	}
	
	/* 连续读取 40 位数据：5 字节 */
	for (i = 0; i < 5; i++)
	{
		for (j = 0; j < 8; j++)
		{
			/* 每位以 50us 低电平开始，等待低电平结束 */
			if (DHT11_WaitLevel(Bit_SET, DHT11_TIMEOUT) == 0)
			{
				DHT11_ErrorCode = 4;	/* 数据位等待超时 */
				goto dht11_error;
			}
			
			/* 高电平持续 26~28us 表示 0，持续 70us 表示 1 */
			Delay_us(40);
			Data[i] <<= 1;
			if (GPIO_ReadInputDataBit(DHT11_GPIO_PORT, DHT11_GPIO_PIN) == Bit_SET)
			{
				Data[i] |= 0x01;
				/* 等待该位的高电平结束 */
				if (DHT11_WaitLevel(Bit_RESET, DHT11_TIMEOUT) == 0)
				{
					DHT11_ErrorCode = 4;
					goto dht11_error;
				}
			}
		}
	}
	
	/* 保存原始数据，校验失败时也能看到读到了什么 */
	for (i = 0; i < 5; i++)
	{
		DHT11_RawData[i] = Data[i];
	}
	
	if (!Primask)
	{
		__enable_irq();
	}
	
	/* 校验：前 4 字节之和的低 8 位等于第 5 字节 */
	if ((uint8_t)(Data[0] + Data[1] + Data[2] + Data[3]) != Data[4])
	{
		DHT11_ErrorCode = 5;
		return 0;
	}
	
	*Humidity = Data[0];		// 湿度整数部分
	*Temperature = Data[2];		// 温度整数部分
	return 1;
	
dht11_error:
	/* 把已经读到的部分数据也保存下来，便于判断时序到了哪一步 */
	for (i = 0; i < 5; i++)
	{
		DHT11_RawData[i] = Data[i];
	}
	if (!Primask)
	{
		__enable_irq();
	}
	return 0;
}
