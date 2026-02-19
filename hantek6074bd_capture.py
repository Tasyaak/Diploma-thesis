import argparse, os, time, ctypes as ct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Iterable, List, Tuple
from ctypes import wintypes


# =========================
# 1) Константы/типы SDK
# =========================

MAX_CH_NUM = 4
MAX_DATA = 255  # в SDK часто используется диапазон 0..255 для "позиционных" величин

WORD = wintypes.WORD   # uint16
UINT = wintypes.UINT   # uint32


class Channel(IntEnum):
    CH1 = 0
    CH2 = 1
    CH3 = 2
    CH4 = 3


class Coupling(IntEnum):
    DC = 0
    AC = 1
    GND = 2


class TriggerMode(IntEnum):
    EDGE = 0


class TriggerSlope(IntEnum):
    RISE = 0
    FALL = 1


class YTFormat(IntEnum):
    NORMAL = 0


class StartControl(IntFlag):
    """
    Управляющие биты для dsoHTStartCollectData:
      bit0: AUTO trigger
      bit1: ROLL mode
      bit2: stop after this collect
    """
    AUTO = 1 << 0
    ROLL = 1 << 1
    STOP_AFTER = 1 << 2


class DeviceState(IntFlag):
    """
    Биты, возвращаемые dsoHTGetState:
      bit0: триггер уже сработал
      bit1: сбор данных завершён
    """
    TRIGGERED = 1 << 0
    DONE = 1 << 1


# Таблица индексов Volt/Div → реальное значение (V/div)
VOLT_DIV_V = {
    0: 2e-3,  1: 5e-3,  2: 10e-3, 3: 20e-3,
    4: 50e-3, 5: 100e-3, 6: 200e-3, 7: 500e-3,
    8: 1.0,   9: 2.0,   10: 5.0,  11: 10.0,
}


# =========================
# 2) Структуры SDK (ctypes.Structure)
# =========================
# В ctypes структуры описываются через наследование от ctypes.Structure и _fields_
# (это стандартный механизм ctypes для C-структур)


class RELAYCONTROL(ct.Structure):
    _fields_ = [
        ("bCHEnable", UINT * MAX_CH_NUM),
        ("nCHVoltDIV", WORD * MAX_CH_NUM),
        ("nCHCoupling", WORD * MAX_CH_NUM),
        ("bCHBWLimit", UINT * MAX_CH_NUM),
        ("nTrigSource", WORD),
        ("bTrigFilt", UINT),
        ("nALT", WORD),
    ]


class CONTROLDATA(ct.Structure):
    _fields_ = [
        ("nCHSet", WORD),
        ("nTimeDIV", WORD),
        ("nTriggerSource", WORD),
        ("nHTriggerPos", WORD),
        ("nVTriggerPos", WORD),
        ("nTriggerSlope", WORD),
        ("nBufferLen", UINT),
        ("nReadDataLen", UINT),
        ("nAlreadyReadLen", UINT),
        ("nALT", WORD),
        ("nETSOpen", WORD),
        ("nDriverCode", WORD),
        ("nLastAddress", UINT),
        ("nFPGAVersion", WORD),
    ]


# =========================
# 3) Dataclass-конфигурация
# =========================

@dataclass(frozen=True)
class ChannelConfig:
    enabled: bool = True
    volt_div_idx: int = 5           # 100 mV/div
    coupling: Coupling = Coupling.DC
    bw_limit: bool = False
    # "вертикальная позиция" (0..255): влияет на смещение кодов в данных
    lever_pos: int = 128            # условно "центр"


@dataclass(frozen=True)
class TriggerConfig:
    source: Channel = Channel.CH1
    mode: TriggerMode = TriggerMode.EDGE
    slope: TriggerSlope = TriggerSlope.RISE
    # уровень триггера тоже в "позиционном" формате 0..255
    level_pos: int = 128


@dataclass(frozen=True)
class CaptureConfig:
    time_div_idx: int = 9           # индекс таймбазы (см. таблицу SDK)
    read_len: int = 0x1000          # 4096 точек
    pretrigger_percent: int = 50    # 0..100
    yt_format: YTFormat = YTFormat.NORMAL
    # режим запуска
    start_control: StartControl = StartControl.AUTO


# =========================
# 4) Исключения и проверки
# =========================

class HantekError(RuntimeError):
    pass


def _check_ok(name : str, ok : int) -> None:
    """
    Большинство функций HTHardDll возвращает 0/1 (FAIL/OK).
    Удобно в одном месте проверять и кидать понятную ошибку.
    """
    if int(ok) == 0:
        raise HantekError(f"{name} failed (returned 0)")


# =========================
# 5) Обёртка над HTHardDll.dll
# =========================

class HantekHardDll:
    """
    Тонкая обёртка над HTHardDll.dll:
    - загружает DLL
    - задаёт argtypes/restype для функций
      (это защищает от неверных типов и помогает ctypes конвертировать аргументы)
    """

    def __init__(self, dll_dir : Path):
        self.dll_dir = dll_dir
        if not dll_dir.exists():
            raise FileNotFoundError(f"dll_dir does not exist: {dll_dir}")

        # Python 3.8+: добавляем путь в DLL-search.
        # Документация прямо говорит, что этот путь используется и ctypes. :contentReference[oaicite:5]{index=5}
        self._dll_handle = os.add_dll_directory(str(dll_dir))

        dll_path = dll_dir / "HTHardDll.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"HTHardDll.dll not found: {dll_path}")

        # WinDLL/windll на Windows использует stdcall-соглашение вызова. :contentReference[oaicite:6]{index=6}
        self.lib = ct.WinDLL(str(dll_path))

        self._bind_prototypes()

    def close(self) -> None:
        # освобождаем добавленный DLL-directory
        try:
            self._dll_handle.close()
        except Exception:
            pass

    def _bind_prototypes(self) -> None:
        L = self.lib

        # connect/find
        L.dsoHTDeviceConnect.argtypes = [WORD]
        L.dsoHTDeviceConnect.restype = WORD

        # init/config
        L.dsoInitHard.argtypes = [WORD]
        L.dsoInitHard.restype = WORD

        L.dsoHTADCCHModGain.argtypes = [WORD, WORD]
        L.dsoHTADCCHModGain.restype = WORD

        L.dsoHTSetSampleRate.argtypes = [WORD, WORD, ct.POINTER(RELAYCONTROL), ct.POINTER(CONTROLDATA)]
        L.dsoHTSetSampleRate.restype = WORD

        L.dsoHTSetCHAndTrigger.argtypes = [WORD, ct.POINTER(RELAYCONTROL), WORD]
        L.dsoHTSetCHAndTrigger.restype = WORD

        L.dsoHTSetRamAndTrigerControl.argtypes = [WORD, WORD, WORD, WORD, WORD]
        L.dsoHTSetRamAndTrigerControl.restype = WORD

        L.dsoHTSetCHPos.argtypes = [WORD, WORD, WORD, WORD, WORD]
        L.dsoHTSetCHPos.restype = WORD

        L.dsoHTSetVTriggerLevel.argtypes = [WORD, WORD, WORD]
        L.dsoHTSetVTriggerLevel.restype = WORD

        L.dsoHTSetTrigerMode.argtypes = [WORD, WORD, WORD, WORD]
        L.dsoHTSetTrigerMode.restype = WORD

        # acquisition
        L.dsoHTStartCollectData.argtypes = [WORD, WORD]
        L.dsoHTStartCollectData.restype = WORD

        L.dsoHTGetState.argtypes = [WORD]
        L.dsoHTGetState.restype = WORD

        L.dsoHTGetData.argtypes = [
            WORD,
            ct.POINTER(WORD), ct.POINTER(WORD), ct.POINTER(WORD), ct.POINTER(WORD),
            ct.POINTER(CONTROLDATA),
        ]
        L.dsoHTGetData.restype = WORD

        # sample rate
        L.dsoGetSampleRate.argtypes = [WORD]
        L.dsoGetSampleRate.restype = ct.c_float

    # --- удобные прокси-методы ---
    def device_connect(self, idx: int) -> bool:
        return bool(self.lib.dsoHTDeviceConnect(WORD(idx)))

    def init_hardware(self, dev: int) -> None:
        _check_ok("dsoInitHard", self.lib.dsoInitHard(WORD(dev)))

    def set_channel_mode_gain(self, dev: int, ch_mode: int) -> None:
        _check_ok("dsoHTADCCHModGain", self.lib.dsoHTADCCHModGain(WORD(dev), WORD(ch_mode)))

    def set_sample_rate(self, dev: int, yt: YTFormat, relay: RELAYCONTROL, ctrl: CONTROLDATA) -> None:
        _check_ok("dsoHTSetSampleRate", self.lib.dsoHTSetSampleRate(WORD(dev), WORD(int(yt)), ct.byref(relay), ct.byref(ctrl)))

    def set_ch_and_trigger(self, dev: int, relay: RELAYCONTROL, time_div_idx: int) -> None:
        _check_ok("dsoHTSetCHAndTrigger", self.lib.dsoHTSetCHAndTrigger(WORD(dev), ct.byref(relay), WORD(time_div_idx)))

    def set_ram_trigger_ctrl(self, dev: int, time_div_idx: int, ch_set_mask: int, trig_src: int, peak: int = 0) -> None:
        _check_ok("dsoHTSetRamAndTrigerControl",
                  self.lib.dsoHTSetRamAndTrigerControl(WORD(dev), WORD(time_div_idx), WORD(ch_set_mask), WORD(trig_src), WORD(peak)))

    def set_ch_pos(self, dev: int, volt_div_idx: int, lever_pos: int, ch: Channel, ch_mode: int) -> None:
        _check_ok("dsoHTSetCHPos", self.lib.dsoHTSetCHPos(WORD(dev), WORD(volt_div_idx), WORD(lever_pos), WORD(int(ch)), WORD(ch_mode)))

    def set_trigger_level(self, dev: int, level_pos: int, ch_mode: int) -> None:
        _check_ok("dsoHTSetVTriggerLevel", self.lib.dsoHTSetVTriggerLevel(WORD(dev), WORD(level_pos), WORD(ch_mode)))

    def set_trigger_mode(self, dev: int, mode: TriggerMode, slope: TriggerSlope, coupling: Coupling) -> None:
        _check_ok("dsoHTSetTrigerMode", self.lib.dsoHTSetTrigerMode(WORD(dev), WORD(int(mode)), WORD(int(slope)), WORD(int(coupling))))

    def start_collect(self, dev: int, start_control: StartControl) -> None:
        _check_ok("dsoHTStartCollectData", self.lib.dsoHTStartCollectData(WORD(dev), WORD(int(start_control))))

    def get_state(self, dev: int) -> DeviceState:
        return DeviceState(int(self.lib.dsoHTGetState(WORD(dev))))

    def get_data(self, dev: int, ch1: ct.Array, ch2: ct.Array, ch3: ct.Array, ch4: ct.Array, ctrl: CONTROLDATA) -> None:
        _check_ok("dsoHTGetData", self.lib.dsoHTGetData(WORD(dev), ch1, ch2, ch3, ch4, ct.byref(ctrl)))

    def get_sample_rate(self, dev: int) -> float:
        return float(self.lib.dsoGetSampleRate(WORD(dev)))


# =========================
# 6) Высокоуровневый класс “осциллограф”
# =========================

class Hantek6074BD:
    """
    Высокоуровневый интерфейс:
      - найти устройство
      - применить конфиг
      - старт/ожидание
      - чтение + конвертация в (t, V)
    """

    def __init__(self, dll_dir : Path):
        self.hard = HantekHardDll(dll_dir)

    def close(self) -> None:
        self.hard.close()

    def find_first_device(self, max_devices: int = 32) -> int:
        for idx in range(max_devices):
            if self.hard.device_connect(idx):
                return idx
        raise HantekError("Device not found: dsoHTDeviceConnect returned 0 for all indices")

    @staticmethod
    def _build_structs(ch_cfgs : List[ChannelConfig],
                       trig_cfg : TriggerConfig,
                       cap_cfg : CaptureConfig) -> Tuple[RELAYCONTROL, CONTROLDATA, int]:
        """
        Собираем RELAYCONTROL + CONTROLDATA из удобных python-конфигов.
        Возвращаем также ch_set_mask.
        """
        relay = RELAYCONTROL()
        ctrl = CONTROLDATA()

        ch_set_mask = 0
        for ch in range(MAX_CH_NUM):
            cfg = ch_cfgs[ch]

            relay.bCHEnable[ch] = 1 if cfg.enabled else 0
            relay.nCHVoltDIV[ch] = WORD(cfg.volt_div_idx)
            relay.nCHCoupling[ch] = WORD(int(cfg.coupling))
            relay.bCHBWLimit[ch] = 1 if cfg.bw_limit else 0

            if cfg.enabled:
                ch_set_mask |= (1 << ch)

        relay.nTrigSource = WORD(int(trig_cfg.source))
        relay.bTrigFilt = 0
        relay.nALT = 0

        ctrl.nCHSet = WORD(ch_set_mask)
        ctrl.nTimeDIV = WORD(cap_cfg.time_div_idx)
        ctrl.nTriggerSource = WORD(int(trig_cfg.source))
        ctrl.nHTriggerPos = WORD(cap_cfg.pretrigger_percent)
        ctrl.nVTriggerPos = WORD(trig_cfg.level_pos)
        ctrl.nTriggerSlope = WORD(int(trig_cfg.slope))
        ctrl.nBufferLen = UINT(cap_cfg.read_len)
        ctrl.nReadDataLen = UINT(cap_cfg.read_len)
        ctrl.nAlreadyReadLen = UINT(0)

        ctrl.nALT = 0
        ctrl.nETSOpen = 0
        ctrl.nDriverCode = 0
        ctrl.nLastAddress = 0
        ctrl.nFPGAVersion = 0

        return relay, ctrl, ch_set_mask

    def configure_and_capture_ch1(
        self,
        dev : int,
        ch1_cfg : ChannelConfig,
        trig_cfg : TriggerConfig,
        cap_cfg : CaptureConfig,
        *,
        ch_mode : int = 4,               # в демо для 4-канальных приборов часто используется 4
        timeout_s : float = 2.0,
        poll_s : float = 0.001,
    ) -> Tuple[List[float], List[float], float]:
        """
        Делает один захват CH1 и возвращает:
          t (сек), v (вольты), fs (Гц)
        """
        # для простоты: CH1 настраиваем, остальные отключаем
        ch_cfgs = [
            ch1_cfg,
            ChannelConfig(enabled=False),
            ChannelConfig(enabled=False),
            ChannelConfig(enabled=False),
        ]

        relay, ctrl, ch_mask = self._build_structs(ch_cfgs, trig_cfg, cap_cfg)

        # 1) init
        self.hard.init_hardware(dev)
        self.hard.set_channel_mode_gain(dev, ch_mode)

        # 2) применить конфигурацию (частота/каналы/триггер/память)
        self.hard.set_sample_rate(dev, cap_cfg.yt_format, relay, ctrl)
        self.hard.set_ch_and_trigger(dev, relay, cap_cfg.time_div_idx)
        self.hard.set_ram_trigger_ctrl(dev, cap_cfg.time_div_idx, ch_mask, int(trig_cfg.source), peak=0)

        # позиции/уровни (часто критично для корректного “центра” и триггера)
        self.hard.set_ch_pos(dev, ch1_cfg.volt_div_idx, ch1_cfg.lever_pos, Channel.CH1, ch_mode)
        self.hard.set_trigger_level(dev, trig_cfg.level_pos, ch_mode)
        self.hard.set_trigger_mode(dev, trig_cfg.mode, trig_cfg.slope, ch1_cfg.coupling)

        # 3) старт
        self.hard.start_collect(dev, cap_cfg.start_control)

        # 4) ожидание окончания (бит DONE)
        t0 = time.time()
        while True:
            st = self.hard.get_state(dev)
            if st & DeviceState.DONE:
                break
            if time.time() - t0 >= timeout_s:
                raise TimeoutError(
                    f"Capture timeout ({timeout_s}s). "
                    f"State={int(st):#x}. Проверьте trigger level/slope/timebase."
                )
            time.sleep(poll_s)

        # 5) чтение буферов
        n = int(ctrl.nReadDataLen)
        ch1 = (WORD * n)()
        ch2 = (WORD * n)()
        ch3 = (WORD * n)()
        ch4 = (WORD * n)()

        self.hard.get_data(dev, ch1, ch2, ch3, ch4, ctrl)

        # 6) фактическая частота дискретизации
        fs = self.hard.get_sample_rate(dev)
        dt = 1.0 / fs if fs > 0 else 0.0

        # 7) конвертация codes -> volts
        # По мануалу/демо для этой серии обычно используется шкала “32 кванта на деление”.
        # Вольты = (counts / 32) * (V/div).
        vdiv = VOLT_DIV_V.get(ch1_cfg.volt_div_idx)
        if vdiv is None:
            raise ValueError(f"Unknown volt_div_idx={ch1_cfg.volt_div_idx}, update VOLT_DIV_V mapping")

        # В данных часто присутствует смещение из-за lever_pos (вертикальной позиции).
        # Здесь используем ту же практику, что в демо: raw - (MAX_DATA - lever_pos)
        y = [ (int(ch1[i]) - (MAX_DATA - ch1_cfg.lever_pos)) / 32.0 * vdiv for i in range(n) ]
        t = [ i * dt for i in range(n) ]

        return t, y, fs


# =========================
# 7) Сохранение + CLI
# =========================

def save_tsv(path : Path, t : Iterable[float], y : Iterable[float], *, fs : float, meta : dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # “человеческий” заголовок с метаданными (удобно для отчёта/проверки)
        f.write("# Hantek 6074BD capture\n")
        f.write(f"# fs_hz={fs:.6f}\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("t_s\tch1_v\n")
        for ti, yi in zip(t, y):
            f.write(f"{ti:.9e}\t{yi:.9e}\n")


DLL_DIR = Path(__file__).resolve().parent / "Hantek_Software" / "Hantek_SDK" / "Dll" / "x64"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="capture_ch1.tsv", help="Выходной файл TSV")
    ap.add_argument("--time-div", type=int, default=9, help="Индекс TimeDiv (из таблицы SDK)")
    ap.add_argument("--volt-div", type=int, default=5, help="Индекс Volt/Div (0..11)")
    ap.add_argument("--len", type=int, default=0x1000, help="Число точек (например 0x1000)")
    ap.add_argument("--timeout", type=float, default=2.0, help="Таймаут ожидания окончания захвата (сек)")
    args = ap.parse_args()

    out = Path(args.out)

    scope = Hantek6074BD(DLL_DIR)
    try:
        dev = scope.find_first_device()

        ch1_cfg = ChannelConfig(
            enabled=True,
            volt_div_idx=args.volt_div,
            coupling=Coupling.DC,
            bw_limit=False,
            lever_pos=128,
        )
        trig_cfg = TriggerConfig(
            source=Channel.CH1,
            mode=TriggerMode.EDGE,
            slope=TriggerSlope.RISE,
            level_pos=128,
        )
        cap_cfg = CaptureConfig(
            time_div_idx=args.time_div,
            read_len=args.len,
            pretrigger_percent=50,
            yt_format=YTFormat.NORMAL,
            start_control=StartControl.AUTO,
        )

        t, y, fs = scope.configure_and_capture_ch1(dev, ch1_cfg, trig_cfg, cap_cfg, timeout_s=args.timeout)

        meta = {
            "device_index": dev,
            "time_div_idx": cap_cfg.time_div_idx,
            "read_len": cap_cfg.read_len,
            "volt_div_idx": ch1_cfg.volt_div_idx,
            "coupling": int(ch1_cfg.coupling),
            "trigger_level_pos": trig_cfg.level_pos,
            "trigger_slope": int(trig_cfg.slope),
        }
        save_tsv(out, t, y, fs=fs, meta=meta)
        print(f"Saved: {out.resolve()}")
        return 0

    finally:
        scope.close()


if __name__ == "__main__":
    raise SystemExit(main())