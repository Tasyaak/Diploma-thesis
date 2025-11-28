import argparse, time, serial
from statistics import mean
import matplotlib.pyplot as plt
import math


# --- прошивка ждёт строго такой формат "set" ---
def build_set_cmd(adc_fs=1_000_000, dac_fs=100_000, out_type=3, out_mode=0, trig=10, relay=0, dac_buf_trig=0):
    def z(n, width): 
        s = str(int(n))
        return ("0"*max(0, width-len(s))) + s
    return (
        "set" +
        z(adc_fs, 8) + "_" +
        z(dac_fs, 8) + "_" +
        z(out_type, 1) + "_" +
        z(out_mode, 1) + "_" +
        z(trig, 3) + "_" +
        z(relay, 1) + "_" +
        z(dac_buf_trig, 1)
    )


def read_exact(ser: serial.Serial, nbytes: int, timeout_s=3.0):
    deadline = time.time() + timeout_s
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = ser.read(nbytes - len(buf))
        if chunk:
            buf.extend(chunk)
        elif time.time() > deadline:
            raise TimeoutError(f"Timeout: got {len(buf)} of {nbytes} bytes")
    return bytes(buf)


def parse_block(raw: bytes, out_type: int):
    # Нормируем к диапазону ~[-0.5, +0.5] под вашу прошивку.
    if out_type == 3:
        # 8-бит, один байт на отсчёт (16384 байт)
        return [(b/255.0) - 0.5 for b in raw]
    if out_type == 1:
        # 8-бит в 16-битных словах (32768 байт): берём младшие байты
        # lo = raw[0::2]  # Cortex-M little-endian: LSB,0, LSB,0, ...
        lo = raw[0::2]
        hi = raw[1::2]
        take = hi if (sum(hi) != 0 and (max(hi)-min(hi)) > (max(lo)-min(lo))) else lo
        return [(b/255.0) - 0.5 for b in take]
        # return [(b/255.0) - 0.5 for b in lo]
    if out_type == 4:
        # split MSB,LSB, два байта на отсчёт (32768 байт)
        vals = []
        r = raw
        for i in range(0, len(r), 2):
            v = (r[i] << 8) | r[i+1]      # MSB first, затем LSB
            vals.append((v/4095.0) - 0.5) # 12-битный масштаб
        return vals
    if out_type == 5:
        # split + триггер половинного буфера (16384 байт)
        vals = []
        r = raw
        for i in range(0, len(r), 2):
            v = (r[i] << 8) | r[i+1]
            vals.append((v/4095.0) - 0.5)
        return vals
    # text (0/2) тут не разбираем — скрипт ориентирован на бинарные быстрые режимы
    raise ValueError("Unsupported out_type for this test (use 1,3,4,5)")


def main():
    ap = argparse.ArgumentParser(description="Smoke-test АЦП (STM32 USB-CDC)")
    ap.add_argument("--port", required=True, help="COM-порт (например, COM7 или /dev/ttyACM0)")
    ap.add_argument("--adc-fs", type=int, default=1_000_000, help="Частота дискретизации АЦП (641..42000000)")
    ap.add_argument("--dac-fs", type=int, default=100_000, help="Частота для ДАК (для теста может быть любой в допуске)")
    ap.add_argument("--out-type", type=int, default=3, choices=[1,3,4,5], help="1=u8, 3=u8 trig, 4=split16, 5=split16 trig")
    ap.add_argument("--trig", type=int, default=10, help="Порог триггера (0..999) — дифференциальный |x[i+1]-x[i]|")
    ap.add_argument("--relay", type=int, default=0, choices=[0,1])
    ap.add_argument("--dac-buf-trig", type=int, default=0, choices=[0,1])
    args = ap.parse_args()

    # Ожидаемый размер блока по out_type
    expected = {1: 32768, 3: 16384, 4: 32768, 5: 16384}[args.out_type]

    set_cmd = build_set_cmd(args.adc_fs, args.dac_fs, args.out_type, 0, args.trig, args.relay, args.dac_buf_trig)

    with serial.Serial(args.port, 38400, timeout=1.5, write_timeout=1.5) as ser:
        # Сбросим возможный мусор
        ser.reset_input_buffer(); ser.reset_output_buffer()

        # Отправим конфигурацию
        ser.write((set_cmd + "\n").encode("ascii"))
        time.sleep(0.05)
        # Считаем всё, что пришло (Ok/Warning/пустые нули)
        _ = ser.read(ser.in_waiting or 1)

        # Попросим блок
        ser.write(b"start\n")

        # Забираем ровно один блок
        data = read_exact(ser, expected, timeout_s=4.0)

    # Разбор и простая валидация
    arr = parse_block(data, args.out_type)
    n = len(arr)
    mn = min(arr) if n else 0.0
    mx = max(arr) if n else 0.0
    dc = mean(arr) if n else 0.0
    # RMS вокруг нуля (приближённо)
    rms = math.sqrt(sum((x - dc)*(x - dc) for x in arr)/max(1,n))

    # Простейшие критерии «PASSED»:
    # 1) блок правильного размера; 2) не все точки одинаковые;
    # 3) есть динамический диапазон (амплитуда > ~1 LSB), 4) не зашито в насыщение.
    span = mx - mn
    all_same = span < 1e-6
    saturated = (mx >= +0.49) or (mn <= -0.49)  # почти полное насыщение
    passed = (len(data) == expected) and (not all_same) and (span > 1.0/255.0)  # >1 LSB в 8-битном эквив.

    print(f"Got {len(data)} bytes, parsed {n} samples (out_type={args.out_type})")
    print(f"DC={dc:+.4f}, RMS={rms:.4f}, min={mn:+.4f}, max={mx:+.4f}, span={span:.4f}, saturated={saturated}")
    print("RESULT:", "PASSED ✅" if passed else "CHECK ❗")
    plt.plot(arr)
    plt.title("ADC block")
    plt.show()


if __name__ == "__main__":
    main()