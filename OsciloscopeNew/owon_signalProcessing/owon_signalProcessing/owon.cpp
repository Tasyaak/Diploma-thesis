#include "pch.h"
#include "owon.h"
#include "common.h"

#pragma comment(lib, "C:/Program Files/IVI Foundation/VISA/Win64/Lib_x64/msc/visa64.lib")

using namespace std;
using namespace std::chrono;

// Вывод матрицы matrix типа type размерности [size1 x size2] в файл под названием outputFileName
template <class type>
void printMatrix(type** matrix, int size1, int size2, string outputFileName) {
	ofstream fout(outputFileName);
	for (int i = 0; i < size1; i++) {
		for (int j = 0; j < size2; j++) {
			fout << matrix[i][j] << " ";
			//fout << abs(matrix[i][j]) << " ";
			//fout << log(abs(matrix[i][j])) << " ";
		}
		fout << endl;
	}
	fout << endl;
	fout.close();
}

// Вывод массива arr типа type размера size в файл под названием outputFileName
template <class type>
void printArray(type* arr, int size, string outputFileName) {
	ofstream fout(outputFileName);
	for (int i = 0; i < size; i++)
		fout << arr[i] << endl;
	fout.close();
}

namespace Complex {
	template <class type>
	struct complex {
		type real;
		type imag;
	};
}

namespace OWON {
	void vertScale(VERTICAL_SCALE sc, float& scCoef, const char*& chSc) {
		switch (sc) {
		case VS2mV:
			chSc = "2mV";
			scCoef = 0.002;
			return;
		case VS5mV:
			chSc = "5mV";
			scCoef = 0.005;
			return;
		case VS10mV:
			chSc = "10mV";
			scCoef = 0.01;
			return;
		case VS20mV:
			chSc = "20mV";
			scCoef = 0.02;
			return;
		case VS50mV:
			chSc = "50mV";
			scCoef = 0.05;
			return;
		case VS100mV:
			chSc = "100mV";
			scCoef = 0.1;
			return;
		case VS200mV:
			chSc = "200mV";
			scCoef = 0.2;
			return;
		case VS500mV:
			chSc = "500mV";
			scCoef = 0.5;
			return;
		case VS1V:
			chSc = "1V";
			scCoef = 1;
			return;
		case VS2V:
			chSc = "2V";
			scCoef = 2;
			return;
		case VS5V:
			chSc = "5V";
			scCoef = 5;
			return;
		default:
			return;
		}
	}

	string vertScale_to_string(VERTICAL_SCALE sc) {
		switch (sc) {
		case VS2mV:
			return "2mV";
		case VS5mV:
			return "5mV";
		case VS10mV:
			return "10mV";
		case VS20mV:
			return "20mV";
		case VS50mV:
			return "50mV";
		case VS100mV:
			return "100mV";
		case VS200mV:
			return "200mV";
		case VS500mV:
			return "500mV";
		case VS1V:
			return "1V";
		case VS2V:
			return "2V";
		case VS5V:
			return "5V";
		default:
			return "";
		}
	}

	string horiScale_to_string(HORIZONTAL_SCALE hs) {
		switch (hs) {
			//case HS1ns:return"1.0ns";
			//case HS2ns:return"2.0ns";
		//case HS5ns:return"5.0ns";
		case HS10ns:return"10ns";
		case HS20ns:return"20ns";
		case HS50ns:return"50ns";
		case HS100ns:return"100ns";
		case HS200ns:return"200ns";
		case HS500ns:return"500ns";
		case HS1us:return"1.0us";
		case HS2us:return"2.0us";
		case HS5us:return"5.0us";
		case HS10us:return"10us";
		case HS20us:return"20us";
		case HS50us:return"50us";
		case HS100us:return"100us";
		case HS200us:return"200us";
		case HS500us:return"500us";
		case HS1ms:return"1.0ms";
		case HS2ms:return"2.0ms";
		case HS5ms:return"5.0ms";
		case HS10ms:return"10ms";
		case HS20ms:return"20ms";
		case HS50ms:return"50ms";
		case HS100ms:return"100ms";
		case HS200ms:return"200ms";
		case HS500ms:return"500ms";
		case HS1s:return"1.0s";
		case HS2s:return"2.0s";
		case HS5s:return"5.0s";
		case HS10s:return"10s";
		case HS20s:return"20s";
		case HS50s:return"50s";
		case HS100s:return"100s";
		default:return"";
		}
	}
	double horiScale_to_number(HORIZONTAL_SCALE hs) {
		switch (hs) {
			//case HS1ns:return"1.0ns";
			//case HS2ns:return"2.0ns";
		case HS5ns:return 5E-9;
		case HS10ns:return 10E-9;
		case HS20ns:return 20E-9;
		case HS50ns:return 50E-9;
		case HS100ns:return 100E-9;
		case HS200ns:return 200E-9;
		case HS500ns:return 500E-9;
		case HS1us:return 1E-6;
		case HS2us:return 2E-6;
		case HS5us:return 5E-6;
		case HS10us:return 10E-6;
		case HS20us:return 20E-6;
		case HS50us:return 50E-6;
		case HS100us:return 100E-6;
		case HS200us:return 200E-6;
		case HS500us:return 500E-6;
		case HS1ms:return 1E-3;
		case HS2ms:return 2E-3;
		case HS5ms:return 5E-3;
		case HS10ms:return 10E-3;
		case HS20ms:return 20E-3;
		case HS50ms:return 50E-3;
		case HS100ms:return 100E-3;
		case HS200ms:return 200E-3;
		case HS500ms:return 500E-3;
		case HS1s:return 1;
		case HS2s:return 0;
		case HS5s:return 5;
		case HS10s:return 10;
		case HS20s:return 20;
		case HS50s:return 50;
		case HS100s:return 100;
		default:return 0;
		}
	}
	HORIZONTAL_SCALE number_to_horiScale(double& hs) {
		if (hs <= 5E-9) {
			hs = 5E-9;
			return HORIZONTAL_SCALE::HS5ns;
		}
		if (hs <= 10E-9) {
			hs = 5E-9;
			return HORIZONTAL_SCALE::HS10ns;
		}
		if (hs <= 20E-9) {
			hs = 20E-9;
			return HORIZONTAL_SCALE::HS20ns;
		}
		if (hs <= 50E-9) {
			hs = 50E-9;
			return HORIZONTAL_SCALE::HS50ns;
		}
		if (hs <= 100E-9) {
			hs = 100E-9;
			return HORIZONTAL_SCALE::HS100ns;
		}
		if (hs <= 200E-9) {
			hs = 200E-9;
			return HORIZONTAL_SCALE::HS200ns;
		}
		if (hs <= 500E-9) {
			hs = 200E-9;
			return HORIZONTAL_SCALE::HS500ns;
		}
		if (hs <= 1E-6) {
			hs = 1E-6;
			return HORIZONTAL_SCALE::HS1us;
		}
		if (hs <= 2E-6) {
			hs = 2E-6;
			return HORIZONTAL_SCALE::HS2us;
		}
		if (hs <= 5E-6) {
			hs = 5E-6;
			return HORIZONTAL_SCALE::HS5us;
		}
		if (hs <= 10E-6) {
			hs = 10E-6;
			return HORIZONTAL_SCALE::HS10us;
		}
		if (hs <= 20E-6) {
			hs = 20E-6;
			return HORIZONTAL_SCALE::HS20us;
		}
		if (hs <= 50E-6) {
			hs = 50E-6;
			return HORIZONTAL_SCALE::HS50us;
		}
		if (hs <= 100E-6) {
			hs = 100E-6;
			return HORIZONTAL_SCALE::HS100us;
		}
		if (hs <= 200E-6) {
			hs = 200E-6;
			return HORIZONTAL_SCALE::HS200us;
		}
		if (hs <= 500E-6) {
			hs = 500E-6;
			return HORIZONTAL_SCALE::HS500us;
		}
		if (hs <= 1E-3) {
			hs = 1E-3;
			return HORIZONTAL_SCALE::HS1ms;
		}
		if (hs <= 2E-3) {
			hs = 1E-3;
			return HORIZONTAL_SCALE::HS2ms;
		}
		if (hs <= 5E-3) {
			hs = 5E-3;
			return HORIZONTAL_SCALE::HS5ms;
		}
		if (hs <= 10E-3) {
			hs = 10E-3;
			return HORIZONTAL_SCALE::HS10ms;
		}
		if (hs <= 20E-3) {
			hs = 20E-3;
			return HORIZONTAL_SCALE::HS20ms;
		}
		if (hs <= 50E-3) {
			hs = 50E-3;
			return HORIZONTAL_SCALE::HS50ms;
		}
		if (hs <= 100E-3) {
			hs = 100E-3;
			return HORIZONTAL_SCALE::HS100ms;
		}
		if (hs <= 200E-3) {
			hs = 200E-3;
			return HORIZONTAL_SCALE::HS200ms;
		}
		if (hs <= 500E-3) {
			hs = 500E-3;
			return HORIZONTAL_SCALE::HS500ms;
		}
		if (hs <= 1) {
			hs = 1;
			return HORIZONTAL_SCALE::HS1s;
		}
		if (hs <= 2) {
			hs = 2;
			return HORIZONTAL_SCALE::HS2s;
		}
		if (hs <= 5) {
			hs = 5;
			return HORIZONTAL_SCALE::HS5s;
		}
		if (hs <= 10) {
			hs = 10;
			return HORIZONTAL_SCALE::HS10s;
		}
		if (hs <= 20) {
			hs = 20;
			return HORIZONTAL_SCALE::HS20s;
		}
		if (hs <= 50) {
			hs = 50;
			return HORIZONTAL_SCALE::HS50s;
		}
		hs = 100;
		return HORIZONTAL_SCALE::HS100s;
	}

	const char* acqDepmem(AQUIRE_DEPMEM ad) {
		switch (ad) {
		case AD1K:
			return "1K";
		case AD10K:
			return "10K";
		case AD100K:
			return "100K";
		case AD1M:
			return "1M";
		case AD10M:
			return "10M";
		case AD100M:
			return "100M";
		default:
			return "";
		}
	}

	const char* coup(COUPLING c) {
		switch (c) {
		case COUPLING_AC:
			return "AC";
		case COUPLING_DC:
			return "DC";
		}
	}

	VERTICAL_SCALE voltageMax_to_vertScale(double& vm) {
		double vs_double = vm / 5;
		if (vs_double <= 0.002) {
			vm = 0.002 * 5;
			return VERTICAL_SCALE::VS2mV;
		}
		if (vs_double <= 0.005) {
			vm = 0.005 * 5;
			return VERTICAL_SCALE::VS5mV;
		}
		if (vs_double <= 0.01) {
			vm = 0.01 * 5;
			return VERTICAL_SCALE::VS10mV;
		}
		if (vs_double <= 0.02) {
			vm = 0.02 * 5;
			return VERTICAL_SCALE::VS20mV;
		}
		if (vs_double <= 0.05) {
			vm = 0.05 * 5;
			return VERTICAL_SCALE::VS50mV;
		}
		if (vs_double <= 0.1) {
			vm = 0.1 * 5;
			return VERTICAL_SCALE::VS100mV;
		}
		if (vs_double <= 0.2) {
			vm = 0.2 * 5;
			return VERTICAL_SCALE::VS200mV;
		}
		if (vs_double <= 0.5) {
			vm = 0.5 * 5;
			return VERTICAL_SCALE::VS500mV;
		}
		if (vs_double <= 1) {
			vm = 1. * 5;
			return VERTICAL_SCALE::VS1V;
		}
		if (vs_double <= 2) {
			vm = 2. * 5;
			return VERTICAL_SCALE::VS2V;
		}
		vm = 5. * 5;
		return VERTICAL_SCALE::VS5V;
	}

	AQUIRE_DEPMEM size_to_acquire(int& sample_size, int& sample_acquire) {
		if (sample_size <= 1E3) {
			sample_acquire = 1E3;
			return AQUIRE_DEPMEM::AD1K;
		}
		if (sample_size <= 10E3) {
			sample_acquire = 10E3;
			return AQUIRE_DEPMEM::AD10K;
		}
		if (sample_size <= 100E3) {
			sample_acquire = 100E3;
			return AQUIRE_DEPMEM::AD100K;
		}
		if (sample_size <= 1E6) {
			sample_acquire = 1E6;
			return AQUIRE_DEPMEM::AD1M;
		}
		sample_acquire = 10E6;
		sample_size = 10E6;
		return AQUIRE_DEPMEM::AD10M;
	}

	string offset_to_string(int offset_size, int sample_size) {
		return to_string(10. * (1 - double(offset_size) / double(sample_size)));
	}

	void doLog(string msg) {
		//FILE* log_file = fopen("OWONLog.txt", "a");
		FILE* log_file; fopen_s(&log_file, "OWONLog.txt", "a");
		fprintf(log_file, "%s", msg.c_str());
		fclose(log_file);
	}

	extern "C" __declspec(dllexport) int getOWONData2(int sample_size, double sample_step, int sample_perSec_max, double voltage_max, int trigger_level, int reading_number, int offset_size, double* result_arr, const char* result_fileName) {
		//FILE* log_file = fopen("OWONLog.txt", "w");
		FILE* log_file; fopen_s(&log_file, "OWONLog.txt", "w");
		fclose(log_file);
		
		int i;

		status1 = viOpenDefaultRM(&defaultRM);
		if (status1 < VI_SUCCESS)
		{
			doLog("Could not open a session to the VISA Resource Manager!\n");
			printf("Could not open a session to the VISA Resource Manager!\n");
			return status1;
		}
		/* Find all the USB TMC VISA resources in our system and store the  */
		/* number of resources in the system in numInstrs.                  */
		ViConstString str = "USB?*INSTR";

		status2 = viFindRsrc(defaultRM, str, &findList, &numInstrs, instrResourceString);

		if (status2 < VI_SUCCESS) {
			doLog("An error occurred while finding resources.\n");
			printf("An error occurred while finding resources.\nHit enter to continue.");
			fflush(stdin);
			viClose(defaultRM);
			return status2;
		}

		/*
		 * Now we will open VISA sessions to all USB TMC instruments.
		 * We must use the handle from viOpenDefaultRM and we must
		 * also use a string that indicates which instrument to open.  This
		 * is called the instrument descriptor.  The format for this string
		 * can be found in the function panel by right clicking on the
		 * descriptor parameter. After opening a session to the
		 * device, we will get a handle to the instrument which we
		 * will use in later VISA functions.  The AccessMode and Timeout
		 * parameters in this function are reserved for future
		 * functionality.  These two parameters are given the value VI_NULL.
		 */
		for (i = 0; i < numInstrs; i++)
		{
			if (i > 0)
				viFindNext(findList, instrResourceString);

			status = viOpen(defaultRM, instrResourceString, VI_EXCLUSIVE_LOCK, VI_NULL, &instr);

			if (status < VI_SUCCESS) {
				doLog("Cannot open a session to the device " + to_string(i + 1) + "\n");
				printf("Cannot open a session to the device %d.\n", i + 1);
				return status;
			}

			int sample_acquired;
			if (sample_perSec_max == 0)
				sample_perSec_max = 125E6;
			AQUIRE_DEPMEM acquiredDepmem = size_to_acquire(sample_size, sample_acquired);
			double horizontalScale_double = sample_acquired * sample_step / 20;
			HORIZONTAL_SCALE horiScale = number_to_horiScale(horizontalScale_double);
			if (1 / (horizontalScale_double * 20 / sample_acquired) > sample_perSec_max) {
				horizontalScale_double = 1. / sample_perSec_max / 20 * sample_acquired;
				horiScale = number_to_horiScale(horizontalScale_double);
			}
			sample_step = horizontalScale_double * 20 / sample_acquired;
			VERTICAL_SCALE vertScale = voltageMax_to_vertScale(voltage_max);

			doLog("Samples per second: " + to_string(int(1 / sample_step)) + "Spsec\n");
			doLog("Max voltage: " + to_string(voltage_max) + "V\n");
			doLog("Sample size: " + to_string(sample_size) + "S\n");
			printf("Samples per second: %d Spsec\n", int(1 / sample_step));
			printf("Max voltage: %f V\n", voltage_max);
			printf("Sample size: %d S\n", sample_size);

			buffer = new char[sample_size * 2 + 16];

			/*
			 * At this point we now have a session open to the USB TMC instrument.
			 * We will now use the viWrite function to send the device the string "*IDN?\n",
			 * asking for the device's identification.
			 */
			string m = (
				string(":*RST;\n") +
				string(":ACQ:PREC 14;\n") +
				string(":ACQ:MODE PEAK;\n") +
				string(":MEAS:TIM 0.002;\n") +
				string(":CH1:SCAL ") + vertScale_to_string(vertScale) + string(";\n") +
				string(":CH1:BAND 20M;\n") +
				string(":CH1:COUP AC;\n") +
				string(":CH1:OFFS 0;\n") +
				string(":CH2:COUP AC;\n") +
				string(":CH2:OFFS 0;\n") +
				string(":HORI:SCAL ") + horiScale_to_string(horiScale) + string(";\n") +
				string(":ACQ:DEPMEM ") + acqDepmem(acquiredDepmem) + string(";\n") +
				string(":TRIG:SING:MODE EDGE;\n") +
				string(":TRIG:SING:EDGE:SOUR CH2;\n") +
				string(":TRIG:SING:EDGE:COUP AC;\n") +
				string(":TRIG:SING:EDGE:SLOP RISE;\n") +
				string(":TRIG:SING:EDGE:LEV ") + to_string(trigger_level) + string(";\n") +
				string(":HORI:OFFS ") + offset_to_string(offset_size, sample_size) + string(";\n") +
				string(":WAV:BEG CH1;\n") +
				string(":WAV:RANG 0,") + to_string(sample_size) + string(";\n") +
				string(":WAV:FETC?;\n") +
				string(":WAV:END;\n")
				);

			const char* st = m.c_str();

			//strcpy(stringinput, st);
			strcpy_s(stringinput, RSIZE_MAX, st);
			
			doLog("Query:\n" + m + '\n');
			printf("\nQuery:\n%s\n", st);

			m = m + m;

			//result_arr = new double[3*sample_size]; // Костыль ;(
			
			int lastNum;
			int ii;
			for (int k = 0; k < reading_number; k++) {
				if (k == 0)
					status = viWrite(instr, (ViBuf)stringinput, (ViUInt32)strlen(stringinput), &writeCount);
				else
					status = viWrite(instr, (ViBuf)":WAV:BEG CH1;\n:WAV:FETC?;\n:WAV:END;\n", (ViUInt32)37, &writeCount);
				if (status < VI_SUCCESS) {
					doLog("Error writing to the device " + to_string(i + 1) + "\n");
					printf("Error writing to the device %d.\n", i + 1);
					status = viClose(instr);
					if (status < VI_SUCCESS) {
						doLog("Error closing the device.\n");
						printf("Error closing the device.\n");
						return status;
					}
					status = viClose(defaultRM);
					if (status < VI_SUCCESS) {
						doLog("Error closing the session.\n");
						printf("Error closing the session.\n");
						return status;
					}
					return status;
				}
				
				lastNum = 0;
				do {
					status = viRead(instr, (ViPBuf)buffer, 2 * sample_size + 16, &retCount);
					if (status < VI_SUCCESS) {
						doLog("Error reading a response from the device " + to_string(i + 1) + "\n");
						printf("Error reading a response from the device %d.\n", i + 1);
						return status;
					}
					
					for (ii = 0; ii < retCount / 2; ii++)
						result_arr[lastNum + ii] += (float(int(buffer[ii * 2])) + int(char(buffer[ii * 2 + 1]) >> 2) * 0.015625) * voltage_max * 0.008; //0.008=1/25 / 5
					
					lastNum += retCount / 2;
				} while (status == VI_SUCCESS_MAX_CNT);
			}

			for (int j = 12; j >= 0; j--)
				result_arr[j] = result_arr[j + 1] + (result_arr[j + 1]-result_arr[j + 2]);

			double div = 1. / reading_number;
			for (int j = 0; j < sample_size; j++)
				result_arr[j] *= div;

			double sumsr = 0;
			for (int j = 0; j < sample_size; j++)
				sumsr += result_arr[j];
			sumsr /= sample_size;
			for (int j = 0; j < sample_size; j++)
				result_arr[j] -= sumsr;

			status = viClose(instr);
			if (status < VI_SUCCESS) {
				doLog("Error closing the device.\n");
				printf("Error closing the device.\n");
				return status;
			}
		}

		/*
		int beta = sample_size / 10;
		for (int t = 0; t < beta; t++)
			result_arr[t] *= 2. * (1. - (0.5 - 0.5 * sin((0.5 * PI * (t - beta)) / beta)));
		for (int t = sample_size - beta; t < sample_size; t++)
			result_arr[t] *= 2. * (0.5 - 0.5 * sin((0.5 * PI * (t - beta)) / beta));
		*/

		if (result_fileName != "") {
			/*
			int size;
			for (int i = 0; ; i++) {
				if (result_fileName[i] == '\0') {
					size = i-1;
					break;
				}
			}

			const char* f2 = new char[size];
			f2 = result_fileName;
			*/

			FILE* result_file; fopen_s(&result_file, result_fileName, "w");

			doLog("Writing data...\n");
			printf("Writing data...\n");
			for (int i = 0; i < sample_size; i++)
				fprintf(result_file, "%f\n", result_arr[i]);

			fclose(result_file);
		}
		//status = viClose(defaultRM); //Crush
		if (status < VI_SUCCESS) {
			doLog("Error closing the session.\n");
			printf("Error closing the session.\n");
			return status;
		}
		else {
			doLog("Success!\n");
			printf("Success!\n");
		}

		return status;
	}
}

namespace Wavelet {
	// Список поддерживаемых вейвлет-функций
	// Численное интегрирование значений функции f с шагом step размера size.
	// Метод интегрирования: Составная формула Симпсона (формула Котеса)
	// Внимание! Если значение интеграла выйдет за пределы допустимого диапазона (будет равен inf), вернётся 0.
	complex<double> kotes(complex<double>* f, double step, int size) {
		complex<double> res = f[0] + f[size - 1];

		for (int k = 1; k < size - 2; k += 2) {
			res += 2. * f[k];
			res += 4. * f[k + 1];
		}

		if (res == res)
			return res * step * 0.33333333333333333;
		else
			return 0;
	}

	double kotes(double* f, double step, int size) {
		double res = f[0] + f[size - 1];

		for (int k = 1; k < size - 2; k += 2) {
			res += 2. * f[k];
			res += 4. * f[k + 1];
		}

		if (res == res)
			return res * step * 0.33333333333333333;
		else
			return 0;
	}

	// Список доступных вейвлет-преобразований.
	// В преобразованиях применяется кратный сдвиг по значению аргумента. Кратность сдвига определялась эмперическим путём, выполняя условие соответствия конечного преобразования частотной оси.
	// Без наличия сдвига итоговый график мог "растягиваться" или "сжиматься" относительно оси частот.
	// При выборе используемого вейвлета рекомендуется применять эрмитов вейвлет (3) в виду наилучшей разрешенности по времени. То есть ему нужно меньше всего временных семплов для схождения значений.
	// (Количество временных семплов важнее частотных, т.к. время работы алгоритма зависит от количества временных семплов квадратично, в то время как от частотных - лишь линейно)
	complex<double> morlet_wavelet(double t) {
		const double sigma = 2 * PI;
		const double kappa = exp(-0.5 * sigma * sigma);
		return (exp(1.0i * sigma * t) - kappa) * exp(-0.5 * t * t);
	}
	double modified_morlet_wavelet(double t) {
		const double sigma = 2 * PI;
		return cos(sigma * t) / cosh(t);
	}
	double hermitian1_wavelet(double t) {
		t *= 4;
		return t * exp(-t * t * 0.5);
	}
	double hermitian2_wavelet(double t) {
		t *= 4;
		return (1 - t * t) * exp(-t * t * 0.5);
	}
	double hermitian3_wavelet(double t) {
		t *= 4;
		return h3cst * (t * t * t - 3 * t) * exp(-t * t * 0.5);
	}
	double hermitian4_wavelet(double t) {
		t *= 4;
		return (t * t * t * t - 6 * t * t + 3) * exp(-t * t * 0.5);
	}
	double poisson2_wavelet(double t) {
		t *= 4;
		/*if (t < 0)
			return 0;
		else
			return (t - 2) * t * exp(-t) * 0.5;*/
		return double(1. - t * t) / double(double(1. + t * t) * double(1. + t * t));
	}

	extern "C" __declspec(dllexport) void sizing_toDouble(
		int t_input_startIndex,
		int t_input_endIndex,
		double t_input_step,
		int t_input_size,
		int t_sizing,
		double* t_sizing_start,
		double* t_sizing_end,
		double* t_sizing_step,
		int* t_sizing_size
	) {
		*t_sizing_start = t_input_startIndex * t_input_step;
		*t_sizing_end = t_input_endIndex * t_input_step;
		*t_sizing_step = t_input_step * t_sizing;
		*t_sizing_size = (t_input_endIndex - t_input_startIndex) / t_sizing;
		return;
	}

	// Определение вейвлет-функции (материнского вейвлета) res вида wavelet для времени от t_start-t_step*t_size до t_start+t_step*t_size с шагом t_step и частот от f_start до f_start+f_step*f_size c шагом f_step.
	// Частота f связана с масштабом s, используемом в оригинальных формулах непрерывного вейвлет-преобразования, соотношением f=1/s.
	// Внимание! В значение вейвлет-функции включен множитель 1/sqrt(f). В оригинальных формулах функций он не представлен, однако так как он всегда используется в прямом и обратном вейвлет-преобразовании, он был включен в саму вейвлет-функцию.
	// Внимание! Материнский вейвлет "отражен" относительно t_start по времени с целью оптимищзации вычислений: вместо того, чтобы иметь тройной массив типа [f_size x t_size x t_size], достаточно иметь массив [f_size x 2*t_size], т.к. tau (параллельный перенос) имеет тот же шаг по значениям, что и время.
	void make_waveletFunction_equalStep(double t_start, double t_step, int t_size, double f_start, double f_step, int f_size, wavelets wavelet, complex<double>* res) {
		for (int i = 0; ; i++)
			cout << i << " " << res[i] << endl;
		switch (wavelet) {
		case hermitian1:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian1_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case hermitian2:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian2_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case hermitian3:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian3_wavelet(t_taus);
					res[i * 2 * t_size + j] *= divsqrt;
				}
			}
			break;
		case hermitian4:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian4_wavelet(t_taus);
					res[i * 2 * t_size + j] *= divsqrt;
				}
			}
			break;
		case poisson2:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = poisson2_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case morlet:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = morlet_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case modified_morlet:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					//res[i][j] = modified_morlet_wavelet(t_taus) * divsqrt;
					cout << i << " " << j << " " << i*2*t_size +j << endl;
					cout << divsqrt << endl;
					res[i * 2*t_size + j] = modified_morlet_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		}
	}
	extern "C" __declspec(dllexport) void make_waveletFunction_equalStep(double t_start, double t_step, int t_size, double f_start, double f_step, int f_size, wavelets wavelet, double* res) {
		switch (wavelet) {
		case hermitian1:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian1_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case hermitian2:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian2_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		case hermitian3:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian3_wavelet(t_taus);
					res[i * 2 * t_size + j] *= divsqrt;
				}
			}
			break;
		case hermitian4:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = hermitian4_wavelet(t_taus);
					res[i * 2 * t_size + j] *= divsqrt;
				}
			}
			break;
		case poisson2:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = poisson2_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		/*case morlet:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					res[i * 2 * t_size + j] = morlet_wavelet(t_taus) * divsqrt;
				}
			}
			break;*/
		case modified_morlet:
			for (int i = 0; i < f_size; i++) {
				double divsqrt = 1 / sqrt(f_start + i * f_step);
				for (int j = 0; j < 2 * t_size; j++) {
					double t_taus = (j - t_size) * t_step * (f_start + i * f_step);
					//res[i][j] = modified_morlet_wavelet(t_taus) * divsqrt;
					res[i * 2*t_size + j] = modified_morlet_wavelet(t_taus) * divsqrt;
				}
			}
			break;
		}
	}

	// Прямое вейвлет-преобразование функции, значения которой взяты из файла inputFileName, имеют временной шаг t_input_step и количество значений t_input_size;
	// В преобразовании используется каждое t_sizing значение, начиная с t_start_index и заканчивая t_end_index;
	// На выходе получается матрица размерности [f_size x t_end_index-t_start_index)/sizing], соответствующие
	// частотам с f_start по f_start+f_step*f_size с шагом f_step
	// и временам с t_input_step*t_start_index по t_input_step*t_end_index с шагом t_input_step*t_sizing.
	// Материнский вейвлет имеет значения psitaus.
	// Значения преобразования записываются в файл outputFile_name.
	// Частота f связана с масштабом s, используемом в оригинальных формулах непрерывного вейвлет-преобразования, соотношением f=1/s.
	// Обратите внимание на особенности массива, хранящего материнский вейвлет! Подробнее в описании функции построения материнского вейвлета make_waveletFunction_equalStep.
	extern "C" __declspec(dllexport) void waveletTransform_fromIndexToIndex(
		char* inputFileName,
		double t_input_step, double t_input_size,
		int t_start_index, int t_end_index, int t_sizing,
		double f_start, double f_step, int f_size,
		double* psitaus,
		char* outputFile_name,
		window win,
		double* win_top_t,
		double* win_top_f,
		int win_top_n,
		double* win_bot_t,
		double* win_bot_f,
		int win_bot_n) {
		double t_start = t_start_index * t_input_step;
		double t_step = t_input_step * t_sizing;
		int t_size = (t_end_index - t_start_index) / t_sizing;

		double** res = new double* [f_size];
		for (int i = 0; i < f_size; i++)
			res[i] = new double[t_size];

		if (t_input_step < 0) {
			cout << "Incorrect input time step!" << endl;
			return;
		}
		if (t_input_size <= 0) {
			cout << "Incorrect input time size!" << endl;
			return;
		}
		if (t_start_index < 0 || t_start_index > t_input_size) {
			cout << "Incorrect time start!" << endl;
			return;
		}
		if (t_end_index <= t_start_index || t_end_index > t_input_size) {
			cout << "Incorrect time end!" << endl;
			return;
		}
		if (t_sizing < 1 || t_sizing > t_input_size) {
			cout << "Incorrect time sizing!" << endl;
			return;
		}
		if (f_start < 0) {
			cout << "Incorrect frequency start!" << endl;
			return;
		}
		if (f_step < 0) {
			cout << "Incorrect frequency step!" << endl;
			return;
		}
		if (f_size <= 0) {
			cout << "Incorrect frequency size!" << endl;
			return;
		}
		
		double* X = new double[t_size];
		ifstream fin(inputFileName);
		double gbg;
		for (int i = 0; i < t_start_index; i++)
			fin >> gbg;
		int read_size = t_end_index - t_start_index;
		double divsizing = 1. / t_sizing;
		for (int i = 0; i < read_size; i++) {
			if (i % t_sizing == 0)
				fin >> X[int(i * divsizing)];
			else fin >> gbg;
		}
		fin.close();
		int t_current_start_index = t_start / t_step; //Номер начального семпла после sizing
		int t_current_end_index = t_current_start_index + t_size; //Номер последнего семпла после sizing

		int nt = omp_get_max_threads();
		omp_set_dynamic(0);
		omp_set_num_threads(nt);
#pragma omp parallel num_threads(nt)
		{
			int tn = omp_get_thread_num();
			complex<double>* f = new complex<double>[t_size];
			for (int i = tn; i < f_size; i += nt)
				for (int j = t_current_start_index; j < t_current_end_index; j++) {
					int temp = t_size - j; // Объяснения о данном сдвиге содержатся в описании функции построения материнского вейвлета make_waveletFunction_equalStep.
					for (int k = t_current_start_index; k < t_current_end_index; k++)
						//f[k - t_current_start_index] = X[k - t_current_start_index] * psitaus[i][temp + k];
						f[k - t_current_start_index] = X[k - t_current_start_index] * psitaus[i * 2 * t_size + temp + k];
					res[i][j - t_current_start_index] = kotes(f, t_step, t_size).real();
				}
			delete[]f;
		}
		delete[]X;

		if (win != window::none) {
			float* t_top_i = new float[win_top_n];
			float* f_top_i = new float[win_top_n];
			for (int i = 0; i < win_top_n; i++) {
				t_top_i[i] = (win_top_t[i] - t_start) / t_step;
				f_top_i[i] = (win_top_f[i] - f_start) / f_step;
			}
			float* t_bot_i = new float[win_bot_n];
			float* f_bot_i = new float[win_bot_n];
			for (int i = 0; i < win_bot_n; i++) {
				t_bot_i[i] = (win_bot_t[i] - t_start) / t_step;
				f_bot_i[i] = (win_bot_f[i] - f_start) / f_step;
			}

			if (win == window::square) {
				for (int i = 0; i < win_top_n - 1; i++) {
					int x0 = t_top_i[i]; int x1 = t_top_i[i + 1];
					int y0 = f_top_i[i]; int y1 = f_top_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						for (int f = cst * float(t) + y0 - cst * float(x0); f < f_size; f++) {
							res[f][t] = 0;
						}
					}
				}
				for (int i = 0; i < win_bot_n - 1; i++) {
					int x0 = t_bot_i[i]; int x1 = t_bot_i[i + 1];
					int y0 = f_bot_i[i]; int y1 = f_bot_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						for (int f = 0; f < cst * float(t) + y0 - cst * float(x0); f++) {
							res[f][t] = 0;
						}
					}
				}
			}
			if (win == window::triangle) {
				int alpha = 10;
				for (int i = 0; i < win_top_n - 1; i++) {
					int x0 = t_top_i[i]; int x1 = t_top_i[i + 1];
					int y0 = f_top_i[i]; int y1 = f_top_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						float f0 = cst * float(t) + y0 - cst * float(x0);
						for (int f = f0 - alpha + 1; f < f_size; f++) {
							if (f < f0 + alpha) {
								res[f][t] *= (f0 + alpha - f) / (2. * alpha);
							}
							else
								res[f][t] = 0;
						}
					}
				}
				for (int i = 0; i < win_bot_n - 1; i++) {
					int x0 = t_bot_i[i]; int x1 = t_bot_i[i + 1];
					int y0 = f_bot_i[i]; int y1 = f_bot_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						float f0 = cst * float(t) + y0 - cst * float(x0);
						for (int f = 0; f < f0 + alpha; f++) {
							if (f < f0 - alpha)
								res[f][t] = 0;
							else
								res[f][t] *= 1 - (f0 - f + alpha) / (2. * alpha);
						}
					}
				}
			}
			if (win == window::RCF) {
				int alpha = 10;
				for (int i = 0; i < win_top_n - 1; i++) {
					int x0 = t_top_i[i]; int x1 = t_top_i[i + 1];
					int y0 = f_top_i[i]; int y1 = f_top_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						float f0 = cst * float(t) + y0 - cst * float(x0);
						for (int f = f0 - alpha + 1; f < f_size; f++) {
							if (f < f0 + alpha) {
								res[f][t] *= (0.5 - 0.5 * sin((0.5 * PI * (f - f0)) / alpha)); //*cos(abs(cst/(1.+cst)));
							}
							else
								res[f][t] = 0;
						}
					}
				}
				for (int i = 0; i < win_bot_n - 1; i++) {
					int x0 = t_bot_i[i]; int x1 = t_bot_i[i + 1];
					int y0 = f_bot_i[i]; int y1 = f_bot_i[i + 1];
					float cst = float(y0 - y1) / float(x0 - x1);
					for (int t = x0; t < x1; t++) {
						float f0 = cst * float(t) + y0 - cst * float(x0);
						for (int f = 0; f < f0 + alpha; f++) {
							if (f < f0 - alpha)
								res[f][t] = 0;
							else
								res[f][t] *= (1 - (0.5 - 0.5 * sin((0.5 * PI * (f - f0)) / alpha))); //*cos(abs(cst/(1.+cst)));
						}
					}
				}
			}
		}

		printMatrix(res, f_size, t_size, outputFile_name);
	}

	// Функция, аналогичная waveletTransform_fromIndexToIndex, но время определяется не начальным и конечным индексами, а значениями в секундах.
	extern "C" __declspec(dllexport) void waveletTransform_fromTimeToTime(
		char* inputFileName,
		double t_input_step, double t_input_size,
		int t_start, int t_end, int t_sizing,
		double f_start, double f_step, int f_size,
		double* psitaus,
		char* outputFile_name,
		window win,
		double* win_top_t,
		double* win_top_f,
		int win_top_n,
		double* win_bot_t,
		double* win_bot_f,
		int win_bot_n) {
		waveletTransform_fromIndexToIndex(
			inputFileName,
			t_input_step, t_input_size,
			t_start/t_input_step, t_end/t_input_step, t_sizing,
			f_start, f_step, f_size,
			psitaus,
			outputFile_name,
			win,
			win_top_t,
			win_top_f,
			win_top_n,
			win_bot_t,
			win_bot_f,
			win_bot_n);
	}

	// Обратное преобразование вейвлет-преобразования, значения которго определены в файле transformedSignal_fileName для значений
	// времени: от t_start до t_start+t_step*t_size с шагом t_step;
	// частот: от f_start до f_start+f_step*f_size с шагом f_step.
	// Материнский вейвлет имеет значения waveletFunction. Обратите внимание на особенности определения вейвлета! Подробнее в описании make_waveletFunction_equalStep.
	// Функция обратного преобразования записывается в файл originalSignal_fileName.
	// Частота f связана с масштабом s, используемом в оригинальных формулах непрерывного вейвлет-преобразования, соотношением f=1/s.
	// Внимание! Полученная функция может иметь кратно большую частоту колебаний, чем должна. Это может быть связано с принципоп неопределённости Гейзенберга.
	// В таком случае рекомендуется либо попробовать использовать другой материнский вейвлет, либо увеличить количество временных семплов в преобразовании (уменьшить t_sizing).
	void backWavelet(char* transformedSignal_fileName,
		double t_start, double t_step, int t_size,
		double f_start, double f_step, int f_size,
		double* waveletFunction,
		char* originalSignal_fileName) {

		double** transformedSignal = new double* [f_size];
		for (int i = 0; i < f_size; i++)
			transformedSignal[i] = new double[t_size];
		ifstream fin(transformedSignal_fileName);
		for (int i = 0; i < f_size; i++)
			for (int j = 0; j < t_size; j++)
				fin >> transformedSignal[i][j];
		fin.close();

		double* res = new double[t_size];

		double* axis_f = new double[f_size];
		for (int i = 0; i < f_size; i++)
			axis_f[i] = f_start + i * f_step;

		int nt = omp_get_max_threads();
		omp_set_dynamic(0);
		omp_set_num_threads(nt);
#pragma omp parallel
		{
			complex<double>* f = new complex<double>[t_size];
			complex<double>* g = new complex<double>[f_size];
			double sqrf;
			int tn = omp_get_thread_num();
			for (int k = tn; k < t_size; k += nt) {
				for (int i = 0; i < f_size; i++) {
					sqrf = axis_f[i] * axis_f[i];
					for (int j = 0; j < t_size; j++)
						f[j] = transformedSignal[i][j] * waveletFunction[i*t_size + t_size + k - j] * sqrf;
					g[i] = kotes(f, t_step, t_size);
				}
				res[k] = kotes(g, f_step, f_size).real();
			}
			delete[] f, g;
		}

		delete[] axis_f;

		printArray(res, t_size, originalSignal_fileName);
	}
}
/*
int main(void){
	int sample_size = 4E3;
	double sample_step = 100E-9; //0 to min
	int sample_perSec_max = 125E6; //0 to default (125E6)
	double voltage_max = 5;
	int reading_number = 100;
	int trigger_level = 0; //from -5 to 5, where 5 equals voltage_max
	double* result_arr = new double[sample_size];
	string result_fileName = "result.txt"; //"" to skip writing
	//#pragma omp parallel for num_threads(THREADS_NUM)
	for (int i = 0; i < sample_size; i++)
		result_arr[i] = 0;

	ViStatus status = OWON::getOWONData2(sample_size, sample_step, sample_perSec_max, voltage_max, trigger_level, reading_number, result_arr, result_fileName);
	if (status == VI_SUCCESS) {
		OWON::doLog("Success!\n");
		printf("Success!\n");
	}
}
*/