#ifndef __AD_H
#define __AD_H

void AD_InitA(void);
void AD_InitB(void);
void Alarm(void);
void deAlarm(void);
uint16_t AD_GetValue(uint8_t ADC_Channel);

#endif
