# DHT11 串口温湿度显示工程

基于 `参考` 工程精简而来，只保留 **DHT11 温湿度采集** 和 **USART1 串口输出**。
OLED、ADC 气体采集、LED、按键、报警等功能均已移除。

## 1. 工程信息

- MCU：STM32F103C8（STM32F103 中容量，64 KB Flash / 20 KB RAM）
- 晶振：8 MHz，系统主频 72 MHz
- 编译器：Keil MDK ARMCC V5.06 update 7
- 工程文件：`DHT11_Serial.uvprojx`
- 输出文件：`Objects/DHT11_Serial.hex`（已勾选生成 HEX）

## 2. 接线

### DHT11

| DHT11 引脚 | STM32 引脚 |
| --- | --- |
| VCC | 3.3 V（DHT11 支持 3.3~5.5 V） |
| GND | GND |
| DATA | **PB5** |

> DATA 与 VCC 之间建议接 **4.7 kΩ ~ 10 kΩ 上拉电阻**。模块自带上拉电阻时可省略。

### 串口（USB-TTL）

| STM32 引脚 | USB-TTL |
| --- | --- |
| PA9（USART1_TX） | RXD |
| GND | GND |

工程只使用发送引脚 PA9，串口助手无需接 PA10。

## 3. 串口参数

- 波特率：**9600**
- 数据位：8
- 停止位：1
- 校验位：无
- 流控：无

## 4. 输出示例

上电后先等待约 1 秒，然后每 2 秒输出一次：

```text
DHT11 serial demo
USART1: PA9(TX), 9600 8N1
DHT11 DATA: PB5

Temperature: 25 C, Humidity: 60 %RH
Temperature: 25 C, Humidity: 61 %RH
```

传感器无响应或校验失败时输出：

```text
DHT11 read error
```

## 5. 使用步骤

1. 用 Keil 打开 `DHT11_Serial.uvprojx`。
2. 点击 `Rebuild` 编译（当前已验证：0 Error，0 Warning）。
3. 连接 ST-Link 下载程序。
4. 用 USB-TTL 连接 PA9/GND，打开串口助手，选择 9600 8N1。
5. 上电或复位，即可看到温湿度数据。

## 6. 修改 DHT11 引脚

打开 `Hardware/DHT11.h`，修改以下三行即可：

```c
#define DHT11_GPIO_PORT     GPIOB
#define DHT11_GPIO_PIN      GPIO_Pin_5
#define DHT11_GPIO_CLK      RCC_APB2Periph_GPIOB
```

例如改到 PA1：

```c
#define DHT11_GPIO_PORT     GPIOA
#define DHT11_GPIO_PIN      GPIO_Pin_1
#define DHT11_GPIO_CLK      RCC_APB2Periph_GPIOA
```

## 7. 文件结构

```text
DHT11_Serial/
├─ Start/       启动文件、CMSIS、系统时钟
├─ Library/     仅保留 GPIO / RCC / USART 标准外设库
├─ System/      SysTick 微秒/毫秒延时
├─ Hardware/
│  ├─ DHT11.c/.h    DHT11 单总线驱动
│  └─ Serial.c/.h   USART1 发送与 printf 重定向
├─ User/
│  ├─ main.c
│  ├─ stm32f10x_conf.h
│  └─ stm32f10x_it.c/.h
└─ DHT11_Serial.uvprojx
```

## 8. 说明

- DHT11 两次采样间隔必须 ≥ 1 秒，本工程设置为 2 秒。
- `Delay_us()` 依赖 72 MHz 系统时钟；若更换主频或晶振，需要同步调整。
- DHT11 只有整数精度：温度 ±2 ℃，湿度 ±5 %RH。
