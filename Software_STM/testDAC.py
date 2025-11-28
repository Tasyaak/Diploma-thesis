import serial
import struct

# Подключение
ser = serial.Serial('COM5', 115200, timeout=1)

# 1. Настройка параметров
ser.write(b'set 2000000 01000000 3 0 100 0 0')
response = ser.read(100)
print(response.decode())  # "Ok\n"

# 2. Загрузка синусоиды в ЦАП
import math
k = 256
for i in range(k):
    value = int(8192 + 8191 * math.sin(2*math.pi * i / k))
    cmd = f'load{i:03d} {value:05d}'.encode()
    ser.write(cmd)
    ser.read(20)  # "Ok [value]\n"

# 3. Запуск захвата
ser.write(b'start')
data = ser.read(16384)  # Бинарный формат 3

# 4. Декодирование данных
samples = struct.unpack('<8192H', data)  # Little-endian uint16
print(f"Получено {len(samples)} отсчетов")
print(f"Минимум: {min(samples)}, Максимум: {max(samples)}")

# 5. Управление реле
ser.write(b'relay_set05 255')
ser.read(20)
ser.write(b'relay_load')
ser.read(10)

ser.close()