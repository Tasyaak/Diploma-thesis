/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "string.h"
#include "usbd_cdc_if.h"
//#include "usbd_cdc_if.c"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define buf_size 16384 //main buffer size
#define buf_size2 32768 //must be twice bigger than previous
#define buf_size3 16384 //buffer for averaging
#define clk_freq 168000000 //pll out frequency
#define relay_number 20 //number of relay boards in stack
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim8;
DMA_HandleTypeDef hdma_tim1_up;
DMA_HandleTypeDef hdma_tim8_up;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
char trans_str[8] = {0,};
char message_str[256] = {0,};
uint8_t testDataToSend[8];
uint8_t out_type = 3;
volatile uint16_t adc[buf_size] = {0,};
volatile uint8_t adc_split[buf_size2] = {0,};
volatile uint16_t average[buf_size3] = {0,};
uint8_t relay_bytes[relay_number] = {0,};
//static char array2[1000] __attribute__((section (".ccmram")));

uint16_t temp_counter;

uint32_t adc_samp_freq = 1000000;
uint32_t dac_samp_freq = 5250000;
_Bool relay_state = 0;
_Bool out_mode = 0;
uint8_t average_counter = 0;
uint8_t trig_level = 0;
volatile uint32_t dac[512] = {0,}; /*{8192, 8393, 8593, 8794, 8994, 9194, 9394, 9592, 9790, 9986, 10182, 10376, 10570, 10761, 10951, 11140, 11326,
		11511, 11694, 11875, 12053, 12229, 12403, 12574, 12743, 12909, 13071, 13231, 13388, 13542, 13693, 13840, 13984, 14125, 14261,
		14395, 14524, 14650, 14771, 14889, 15003, 15113, 15218, 15319, 15416, 15509, 15597, 15681, 15760, 15835, 15905, 15970, 16031,
		16087, 16138, 16184, 16226, 16263, 16295, 16322, 16344, 16361, 16374, 16381, 16383, 16381, 16374, 16361, 16344, 16322, 16295,
		16263, 16226, 16184, 16138, 16087, 16031, 15970, 15905, 15835, 15760, 15681, 15597, 15509, 15416, 15319, 15218, 15113, 15003,
		14889, 14771, 14650, 14524, 14395, 14261, 14125, 13984, 13840, 13693, 13542, 13388, 13231, 13071, 12909, 12743, 12574, 12403,
		12229, 12053, 11875, 11694, 11511, 11326, 11140, 10951, 10761, 10570, 10376, 10182, 9986, 9790, 9592, 9394, 9194, 8994, 8794,
		8593, 8393, 8192, 7990, 7790, 7589, 7389, 7189, 6989, 6791, 6593, 6397, 6201, 6007, 5813, 5622, 5432, 5243, 5057, 4872, 4689,
		4508, 4330, 4154, 3980, 3809, 3640, 3474, 3312, 3152, 2995, 2841, 2690, 2543, 2399, 2258, 2122, 1988, 1859, 1733, 1612, 1494,
		1380, 1270, 1165, 1064, 967, 874, 786, 702, 623, 548, 478, 413, 352, 296, 245, 199, 157, 120, 88, 61, 39, 22, 9, 2, 1, 2, 9, 22,
		39, 61, 88, 120, 157, 199, 245, 296, 352, 413, 478, 548, 623, 702, 786, 874, 967, 1064, 1165, 1270, 1380, 1494, 1612, 1733, 1859,
		1988, 2122, 2258, 2399, 2543, 2690, 2841, 2995, 3152, 3312, 3474, 3640, 3809, 3980, 4154, 4330, 4508, 4689, 4872, 5057, 5243,
		5432, 5622, 5813, 6007, 6201, 6397, 6593, 6791, 6989, 7189, 7389, 7589, 7790, 7990};*/ //sin array
volatile uint8_t flag = 1;
volatile uint8_t dac_flag = 1;
_Bool dac_buf_trig = 0;
uint8_t sync_flag = 0;
uint16_t prev_value = 0;
uint16_t current_value = 0;
volatile uint8_t it_on = 1;
char receive_buf[516] = {0};
uint8_t count = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM8_Init(void);
static void MX_SPI1_Init(void);
/* USER CODE BEGIN PFP */
void Delay(uint16_t ticks) {
	for(uint16_t i = 0; i < ticks; i++) {
		__ASM("NOP");
	}
}

_Bool Compare(char* a, char* b) {
	for (int i = 0; i < strlen(b); i++) {
		if (a[i] != b[i]) {
			return 0;
		}
	}
	return 1;
}

uint32_t myAtoi(char* str)
{
	uint32_t res = 0;

    for (int i = 0; str[i] != '\0'; ++i)
        res = res * 10 + str[i] - '0';

    return res;
}

char* cut(char* str, int start, int end) {
	static char outstr[100] = {0};
	for (int i = start; i < end; i++) {
		outstr[i - start] = str[i];
	}
	outstr[end - start] = '\0';
	return outstr;
}

void clear(char* str) {
	for (int i = 0; i < 256; i++) {
		str[i] = 0;
	}
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM1_Init();
  MX_USART1_UART_Init();
  MX_USB_DEVICE_Init();
  MX_TIM2_Init();
  MX_TIM4_Init();
  MX_TIM8_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */

  for (int i = 0; i < 256; i++) {
	  dac[i] = 8192;
  }

  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
  HAL_SPI_Transmit(&hspi1, relay_bytes, 20, 1000);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);

  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  TIM1->PSC = 41;
  TIM1->CCR1=1;


  HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
  TIM2->PSC = 3;
  TIM2->CCR1=1;

  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1);
  TIM8->PSC = 3;
  TIM8->CCR1=1;
/*
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_1);
  TIM4->PSC = 211;
  TIM1->CCR1=1;*/

  HAL_TIM_Base_Start(&htim1);
  HAL_DMA_Start_IT(&hdma_tim1_up,(uint32_t)&GPIOB->IDR,(uint32_t)adc,buf_size);
  __HAL_TIM_ENABLE_DMA(&htim1, TIM_DMA_UPDATE);

  HAL_TIM_Base_Start(&htim8);
  HAL_DMA_Start_IT(&hdma_tim8_up,(uint32_t)dac,(uint32_t)&GPIOC->ODR,512);
  __HAL_TIM_ENABLE_DMA(&htim8, TIM_DMA_UPDATE);

  //__HAL_TIM_ENABLE(&htim2);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	 // count++;
	  //GPIOC->ODR = dac[count];
	  //testDataToSend[0] = 5;
	  //if (dac_flag == 1) {
	  		  //dac_flag = 0;
	  		  //HAL_DMA_Start_IT(&hdma_tim2_up_ch3,(uint32_t)dac, (uint32_t)&GPIOC->ODR, 256);
	  	//  }
	  //CDC_Receive_FS(receive_buf, 6);
	  if ((flag == 1 && (!dac_buf_trig)) || (dac_flag == 1 && dac_buf_trig)) {
		  it_on = 0;
		  flag = 0;
		  dac_flag = 0;
		  if (Compare(receive_buf, "start") || out_mode) {
			  if (out_type == 0) {
					  HAL_Delay(100);
					  for(int i = 0; i < buf_size; i++) {
						  snprintf(trans_str, 7, "%d\n", adc[i]);
						  CDC_Transmit_FS((uint8_t*)trans_str, 8);
						  Delay(10000);
					  }
			  }

			  else if(out_type == 1) {
				  for (int i = 0; i < buf_size; i++) {
					  adc[i] = adc[i]/16;
				  }
				  CDC_Transmit_FS((uint8_t*)adc, buf_size*2);
				  HAL_Delay(200);
			  }

			  else if(out_type == 2) {
				  int start_point = 0;
				  HAL_Delay(100);
				  for (int i = 0; i < buf_size/2; i++) {
					  if (abs(adc[i+1] - adc[i]) > trig_level) {
						  start_point = i;
					  }
				  }
				  for(int i = start_point; i < start_point + buf_size/2; i++) {
					  snprintf(trans_str, 6, "%d\n", adc[i]);
					  CDC_Transmit_FS((uint8_t*)trans_str, 6);
					  Delay(10000);
				  }
			  }

			  else if(out_type == 3) {
				  int start_point = 0;
				  for (int i = 0; i < buf_size/2; i++) {
					  if (abs(adc[i+1] - adc[i]) > trig_level) {
						  start_point = i;
					  }
				  }
				  for(int i = start_point; i < start_point + buf_size/2; i++) {
					  adc[i - start_point] = adc[i]/16;
				  }
				  CDC_Transmit_FS((uint8_t*)adc, buf_size);
				  HAL_Delay(200); // ?

			  }

			  else if(out_type == 4) {
				  for(int i = 0; i < buf_size; i++) {
					  adc_split[i*2] = ((adc[i] & 0xFF00) >> 8);
					  adc_split[i*2 + 1] = ((adc[i] & 0x00FF));

				  }
				  CDC_Transmit_FS((uint8_t*)adc_split, 2*buf_size);
				  HAL_Delay(200);
			  }

			  else if(out_type == 5) {
				  int start_point = 0;
				  for (int i = 0; i < buf_size/2; i++) {
					  if (abs(adc[i+1] - adc[i]) > trig_level) {
						  start_point = i;
					  }
				  }

				  for(int i = start_point; i < start_point + buf_size/2; i++) {
					  adc_split[(i - start_point)*2] = ((adc[i] & 0xFF00) >> 8);
					  adc_split[(i - start_point)*2 + 1] = ((adc[i] & 0x00FF));

				  }
				  CDC_Transmit_FS((uint8_t*)adc_split, buf_size);
				  HAL_Delay(200);
			  }

			  clear(receive_buf);
		  }
		  else if (Compare(receive_buf, "average")) {
			  clear(message_str);
			  snprintf(message_str, 256, "This mode is currently incompatible with set parameters\n");
			  CDC_Transmit_FS((uint8_t*)message_str, 57);
			  clear(message_str);
			  clear(receive_buf);
		  }

		  sync_flag = 1;

		  HAL_DMA_Start_IT(&hdma_tim1_up,(uint32_t)&GPIOB->IDR,(uint32_t)adc,buf_size);
		  it_on = 1;
	  }

	  if (Compare(receive_buf, "load")) {
		  uint16_t temp_index = (uint16_t) myAtoi(cut(receive_buf, 4, 7));
		  uint16_t temp_value = (uint16_t) myAtoi(cut(receive_buf, 8, 13));
		  dac[temp_index] = abs(temp_value - 16383);
		  snprintf(message_str, 256, "Ok %d \n", temp_value);
		  CDC_Transmit_FS((uint8_t*)message_str, 20);
		  clear(receive_buf);
		  clear(message_str);
	  }

	  else if (Compare(receive_buf, "relay_set")) {
		  uint8_t temp_index = (uint8_t) myAtoi(cut(receive_buf, 9, 11));
		  uint8_t temp_value = (uint8_t) myAtoi(cut(receive_buf, 12, 15));
		  relay_bytes[(relay_number - 1) - temp_index] = temp_value;
		  snprintf(message_str, 256, "Ok %d \n", temp_value);
		  CDC_Transmit_FS((uint8_t*)message_str, 20);
		  clear(receive_buf);
		  clear(message_str);
	  }

	  else if (Compare(receive_buf, "relay_load")) {
		  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
		  HAL_SPI_Transmit(&hspi1, relay_bytes, 20, 1000);
		  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);

		  snprintf(message_str, 256, "Ok\n");
		  CDC_Transmit_FS((uint8_t*)message_str, 4);
		  clear(receive_buf);
		  clear(message_str);
	  }

	  else if (Compare(receive_buf, "current_config")) {
		  snprintf(message_str, 256, "ADC sampling frequency: %ld Hz\nDAC sampling frequency: %ld Hz\nOut data format: %d\nOut mode: %d\nTrigger edge level: %d\nRelay: %d\nTrigger by half DMA DAC transfer complete flag: %d\n",
				  adc_samp_freq, dac_samp_freq, out_type, out_mode, trig_level, relay_state, dac_buf_trig);
		  CDC_Transmit_FS((uint8_t*)message_str, 256);
		  clear(receive_buf);
		  clear(message_str);
	  }

	  else if (Compare(receive_buf, "set")) {
		  adc_samp_freq = myAtoi(cut(receive_buf, 3, 11));
		  dac_samp_freq = myAtoi(cut(receive_buf, 12, 20));
		  uint8_t temp_out_type = myAtoi(cut(receive_buf, 21, 22));
		  out_mode = (_Bool)myAtoi(cut(receive_buf, 23, 24));
		  trig_level = (uint8_t)myAtoi(cut(receive_buf, 25, 28));
		  relay_state = (_Bool)myAtoi(cut(receive_buf, 29, 30));
		  dac_buf_trig = (_Bool)myAtoi(cut(receive_buf, 31, 32));
		  if (adc_samp_freq > 42000000 || adc_samp_freq < 641) {
			  adc_samp_freq = 1000000;
			  clear(message_str);
			  snprintf(message_str, 256, "Warning! Invalid ADC sampling frequency. Set to default: %ld \n", adc_samp_freq);
			  CDC_Transmit_FS((uint8_t*)message_str, 70);
			  clear(message_str);
			  HAL_Delay(10);

		  }

		  if (dac_samp_freq > 42000000 || dac_samp_freq < 641) {
			  dac_samp_freq = 100000;
			  clear(message_str);
			  snprintf(message_str, 256, "Warning! Invalid DAC sampling frequency. Set to default: %ld \n", dac_samp_freq);
			  CDC_Transmit_FS((uint8_t*)message_str, 70);
			  clear(message_str);
			  HAL_Delay(10);

		  }

		  if (temp_out_type > 5) {
			  temp_out_type = 3;
			  clear(message_str);
			  snprintf(message_str, 256, "Warning! Invalid out type. Set to default: %d \n", temp_out_type);
			  CDC_Transmit_FS((uint8_t*)message_str, 47);
			  clear(message_str);
			  HAL_Delay(10);
		  }

		  TIM1->PSC = (((clk_freq/4)/adc_samp_freq) - 1);
		  uint16_t dac_presc = (((clk_freq/8)/dac_samp_freq) - 1);
		  TIM2->PSC = dac_presc;
		  TIM8->PSC = dac_presc;
		  out_type = temp_out_type;
		  if (relay_state) {
			  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_2, GPIO_PIN_SET);
		  }
		  else {
			  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_2, GPIO_PIN_RESET);
		  }
		  snprintf(message_str, 256, "Ok\n");
		  CDC_Transmit_FS((uint8_t*)message_str, 4);
		  clear(receive_buf);
		  clear(message_str);

	  }




    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_128;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 4;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 3;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV2;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 420;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 3;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV2;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_ENABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 420;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 255;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV2;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 420;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 3;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim8) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim8, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim8, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMA2_Stream5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream5_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_2, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13|GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2
                          |GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6
                          |GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10
                          |GPIO_PIN_11|GPIO_PIN_12, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);

  /*Configure GPIO pin : PE2 */
  GPIO_InitStruct.Pin = GPIO_PIN_2;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : PC13 PC0 PC1 PC2
                           PC3 PC4 PC5 PC6
                           PC7 PC8 PC9 PC10
                           PC11 PC12 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2
                          |GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6
                          |GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10
                          |GPIO_PIN_11|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PA4 */
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 PB2 PB10
                           PB11 PB3 PB4 PB5
                           PB6 PB7 PB8 PB9 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_10
                          |GPIO_PIN_11|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5
                          |GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PB12 PB13 PB14 PB15 */
  GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
