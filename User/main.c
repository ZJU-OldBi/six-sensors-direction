#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "AD.h"
#include "Serial.h"

uint16_t VOC1;
uint16_t VOC2,VOC3,VOC4,VOC5,VOC6;
uint8_t RxData;
uint8_t Array[] = {0,0,0};

float VOCF1;
float VOCF2,VOCF3,VOCF4,VOCF5,VOCF6;

// 新增变量
uint16_t baseline1, baseline2, baseline3, baseline4, baseline5, baseline6;
uint16_t threshold1, threshold2, threshold3, threshold4, threshold5, threshold6;
uint8_t system_state = 0; // 0: 预热中, 1: 采集基准值, 2: 正常监测

// 初始化B0端口LED
void LEDup_Init(void)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
	
	// 初始状态关闭LED
	GPIO_SetBits(GPIOB, GPIO_Pin_0);
}

// 控制LED
void LED_Control(uint8_t state)
{
	if (state)
	{
		GPIO_ResetBits(GPIOB, GPIO_Pin_0); // 点亮LED
	}
	else
	{
		GPIO_SetBits(GPIOB, GPIO_Pin_0); // 关闭LED
	}
}

int main(void)
{
	OLED_Init();
	AD_InitA();
	AD_InitB();
	Serial_Init();
	LEDup_Init();
	Alarm();
	
	// 预热30秒
	Serial_Printf("System warming up...\r\n");
	for (int i = 0; i < 30; i++) // 3秒 = 30 * 100ms
	{
		Delay_ms(100);
	}
	Serial_Printf("Warming up completed!\r\n");
	
	// 采集5秒内的平均值作为基准值
	Serial_Printf("Collecting baseline data...\r\n");
	uint32_t sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0;
	for (int i = 0; i < 50; i++) // 5秒 = 50 * 100ms
	{
		VOC1 = AD_GetValue(ADC_Channel_0);
		VOC2 = AD_GetValue(ADC_Channel_1);
		VOC3 = AD_GetValue(ADC_Channel_2);
		VOC4 = AD_GetValue(ADC_Channel_3);
		VOC5 = AD_GetValue(ADC_Channel_4);
		VOC6 = AD_GetValue(ADC_Channel_5);
		
		sum1 += VOC1;
		sum2 += VOC2;
		sum3 += VOC3;
		sum4 += VOC4;
		sum5 += VOC5;
		sum6 += VOC6;
		
		Delay_ms(100);
	}
	
	// 计算基准值
	baseline1 = sum1 / 50;
	baseline2 = sum2 / 50;
	baseline3 = sum3 / 50;
	baseline4 = sum4 / 50;
	baseline5 = sum5 / 50;
	baseline6 = sum6 / 50;
	
	// 计算报警阈值
	threshold1 = baseline1 + 200;
	threshold2 = baseline2 + 200;
	threshold3 = baseline3 + 200;
	threshold4 = baseline4 + 200;
	threshold5 = baseline5 + 200;
	threshold6 = baseline6 + 200;
	
	// 输出基准值和阈值
	Serial_Printf("Baseline values: %d, %d, %d, %d, %d, %d\r\n", baseline1, baseline2, baseline3, baseline4, baseline5, baseline6);
	Serial_Printf("Threshold values: %d, %d, %d, %d, %d, %d\r\n", threshold1, threshold2, threshold3, threshold4, threshold5, threshold6);
	Serial_Printf("System ready for monitoring!\r\n");
	
	// 正常监测模式
	while (1)
	{
		VOC1 = AD_GetValue(ADC_Channel_0);
		VOC2 = AD_GetValue(ADC_Channel_1);
		VOC3 = AD_GetValue(ADC_Channel_2);
		VOC4 = AD_GetValue(ADC_Channel_3);
		VOC5 = AD_GetValue(ADC_Channel_4);
		VOC6 = AD_GetValue(ADC_Channel_5);
		
		// 检查是否超过阈值
		uint8_t alarm = 0;
		uint8_t alarm_channel = 0;
		if (VOC1 > threshold1)
		{
			alarm = 1;
			alarm_channel = 1;
		}
		if (VOC2 > threshold2)
		{
			alarm = 1;
			alarm_channel = 2;
		}
		if (VOC3 > threshold3)
		{
			alarm = 1;
			alarm_channel = 3;
		}
		if (VOC4 > threshold4)
		{
			alarm = 1;
			alarm_channel = 4;
		}
		if (VOC5 > threshold5)
		{
			alarm = 1;
			alarm_channel = 5;
		}
		if (VOC6 > threshold6)
		{
			alarm = 1;
			alarm_channel = 6;
		}
		
		if (alarm)
		{
			LED_Control(1); // 点亮LED报警
			Alarm(); // 调用现有报警函数
			Serial_Printf("ALARM: Channel %d exceeded threshold!\r\n", alarm_channel);
		}
		else
		{
			LED_Control(0); // 关闭LED
			deAlarm(); // 调用现有取消报警函数
		}
		
		// 发送数据到串口
		Serial_Printf("%d,%d,%d,%d,%d,%d,%d,%d\r\n", VOC1, VOC2, VOC3, VOC4, VOC5, VOC6, alarm, alarm_channel);
		
		Delay_ms(100);
	}
}
