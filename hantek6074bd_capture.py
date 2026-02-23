import os, time, ctypes as ct, numpy as np
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Tuple, Optional, Literal
from ctypes import wintypes


# =========================
# Константы/типы SDK
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


# Таблица индексов Volt/Div → реальное значение (V/div) из мануала SDK
VOLT_DIV_V = {
    0: 2e-3,  1: 5e-3,  2: 10e-3, 3: 20e-3,
    4: 50e-3, 5: 100e-3, 6: 200e-3, 7: 500e-3,
    8: 1.0,   9: 2.0,   10: 5.0,  11: 10.0,
}


# =========================
# Структуры SDK (ctypes.Structure)
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
        ("nHTriggerPos", WORD),     # pretrigger %, 0..100
        ("nVTriggerPos", WORD),     # trigger level pos, 0..255 (условно)
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


# -------------------------
# Параметры захвата (CH1-only)
# -------------------------

@dataclass(frozen=True)
class ScanParams:
    # waveform window
    time_div_idx : int = 9
    read_len : int = 0x1000
    pretrigger_percent : int = 50
    yt_format : YTFormat = YTFormat.NORMAL

    # channel CH1
    volt_div_idx : int = 5               # 100 mV/div
    coupling : Coupling = Coupling.DC
    bw_limit : bool = False
    lever_pos : int = 128                # вертикальная позиция 0..255, условно центр

    # trigger
    trig_source : Channel = Channel.CH1
    trig_mode : TriggerMode = TriggerMode.EDGE
    trig_slope : TriggerSlope = TriggerSlope.RISE
    trig_level_pos : int = 128

    # acquisition control
    start_control : StartControl = StartControl.AUTO    # режим запуска
    timeout_s : float = 2.0
    poll_s : float = 0.001


@dataclass(frozen=True)
class AScanFrame:
    """
    t и raw могут быть view на внутренний буфер (если copy=False), при следующем capture() данные будут перезаписаны
    t — кэшируемый массив времени (не меняется при фиксированных параметрах)
    """
    point : Optional[Tuple[float, float]]
    fs_hz : float
    dt_s : float
    t_s : np.ndarray
    triggered : bool
    volts : Optional[np.ndarray] = None
    raw_u16 : Optional[np.ndarray] = None


# =========================
# Исключения и проверки
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
# Обёртка над HTHardDll.dll
# =========================

class HantekHardDll:
    """
    Тонкая обёртка над HTHardDll.dll
    """
    def __init__(self, dll_dir : Path, *, channel_mode : int = 4, device_index : Optional[int] = None, max_devices : int = 32) -> None:
        self.channel_mode = int(channel_mode)
        self.max_devices = int(max_devices)

        # Python 3.8+: добавляем путь в DLL-search
        # Документация прямо говорит, что этот путь используется и ctypes
        self._dll_cookie = os.add_dll_directory(str(dll_dir))

        dll_path = dll_dir / "HTHardDll.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"HTHardDll.dll not found: {dll_path}")

        # WinDLL/windll на Windows использует stdcall-соглашение вызова
        self.lib = ct.WinDLL(str(dll_path))
        self._bind_prototypes()
        self.device_index = device_index if device_index is not None else self._find_device()

        # init один раз
        _check_ok("dsoInitHard", self.lib.dsoInitHard(WORD(self.device_index)))
        _check_ok("dsoHTADCCHModGain", self.lib.dsoHTADCCHModGain(WORD(self.device_index), WORD(self.channel_mode)))

        # Будет заполнено при configure()
        self.params : Optional[ScanParams] = None
        self.fs_hz : float = 0.0
        self.dt_s : float = 0.0
        
        # Структуры управления — живут весь скан
        self._relay = RELAYCONTROL()
        self._ctrl = CONTROLDATA()

        # Буферы данных (4 канала) — выделим при configure()
        self._buf_len : int = 0
        self._ch1_buf : Optional[ct.Array[WORD]] = None
        self._ch2_buf : Optional[ct.Array[WORD]] = None
        self._ch3_buf : Optional[ct.Array[WORD]] = None
        self._ch4_buf : Optional[ct.Array[WORD]] = None

        # Кэш времени (при фиксированных параметрах строится один раз)
        self._t_cache : Optional[np.ndarray] = None

    def close(self) -> None:
        # освобождаем добавленный DLL-directory
        try:
            self._dll_cookie.close()
        except Exception:
            pass
    
    def __enter__(self) -> "HantekHardDll":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
    
    def configure(self, params : ScanParams) -> None:
        """
        Применить параметры к прибору 1 раз перед сканом
        После этого можно вызывать capture(point) сколько угодно раз
        Повторный configure() допустим между сканами
        """

        self.params = params
        self._ensure_buffers(params.read_len)
        self._fill_structs(params)

        # Основные "тяжёлые" вызовы конфигурации
        _check_ok("dsoHTSetSampleRate",
            self.lib.dsoHTSetSampleRate(
                WORD(self.device_index),
                WORD(int(params.yt_format)),
                ct.byref(self._relay),
                ct.byref(self._ctrl),
            ))

        _check_ok("dsoHTSetCHAndTrigger",
            self.lib.dsoHTSetCHAndTrigger(
                WORD(self.device_index),
                ct.byref(self._relay),
                WORD(params.time_div_idx),
            ))

        _check_ok("dsoHTSetRamAndTrigerControl",
            self.lib.dsoHTSetRamAndTrigerControl(
                WORD(self.device_index),
                WORD(params.time_div_idx),
                WORD(int(self._ctrl.nCHSet)),
                WORD(int(params.trig_source)),
                WORD(0),
            ))

        # “лёгкие” настройки
        _check_ok("dsoHTSetCHPos",
            self.lib.dsoHTSetCHPos(
                WORD(self.device_index),
                WORD(params.volt_div_idx),
                WORD(params.lever_pos),
                WORD(int(Channel.CH1)),
                WORD(self.channel_mode),
            ))

        _check_ok("dsoHTSetVTriggerLevel",
            self.lib.dsoHTSetVTriggerLevel(
                WORD(self.device_index),
                WORD(params.trig_level_pos),
                WORD(self.channel_mode),
            ))

        _check_ok("dsoHTSetTrigerMode",
            self.lib.dsoHTSetTrigerMode(
                WORD(self.device_index),
                WORD(int(params.trig_mode)),
                WORD(int(params.trig_slope)),
                WORD(int(params.coupling)),
            ))

        # Фактическая fs (после set_sample_rate)
        self.fs_hz = float(self.lib.dsoGetSampleRate(WORD(self.device_index)))
        self.dt_s = 1.0 / self.fs_hz if self.fs_hz > 0 else 0.0

        if self.fs_hz <= 0:
            raise HantekError("Sample rate is zero after configure(); check time_div_idx / device state")
        
        # Кэш оси времени
        self._t_cache = self._build_time_axis(
            n=params.read_len,
            dt=self.dt_s,
            pretrigger_percent=params.pretrigger_percent,
        )

    def capture(
        self,
        *,
        point : Optional[Tuple[float, float]] = None,
        return_mode : Literal["raw", "volts", "both"] = "volts",
        copy : bool = False,
    ) -> AScanFrame:
        """
        Один захват A-scan в текущей точке (CH1)

        return_mode:
          - "volts": вернуть только volts (экономия памяти)
          - "raw": вернуть только raw_u16
          - "both": вернуть volts + raw_u16

        copy:
          - False (по умолчанию): максимальная скорость, данные = view на внутренний буфер
          - True: массивы копируются и безопасны для хранения
        """
        if self.params is None or self._t_cache is None:
            raise RuntimeError("Call configure(params) before capture().")

        p = self.params
        # старт
        _check_ok("dsoHTStartCollectData",
            self.lib.dsoHTStartCollectData(WORD(self.device_index), WORD(int(p.start_control))))

        # ожидание DONE, параллельно отмечаем TRIGGERED
        triggered = False
        t0 = time.time()
        while True:
            st = DeviceState(int(self.lib.dsoHTGetState(WORD(self.device_index))))
            if st & DeviceState.TRIGGERED:
                triggered = True
            if st & DeviceState.DONE:
                break
            if time.time() - t0 >= p.timeout_s:
                raise TimeoutError(f"Capture timeout {p.timeout_s}s; state={int(st):#x}")
            time.sleep(p.poll_s)

        # чтение данных в буферы
        assert self._ch1_buf is not None
        assert self._ch2_buf is not None
        assert self._ch3_buf is not None
        assert self._ch4_buf is not None

        _check_ok("dsoHTGetData",
            self.lib.dsoHTGetData(
                WORD(self.device_index),
                self._ch1_buf, self._ch2_buf, self._ch3_buf, self._ch4_buf,
                ct.byref(self._ctrl),
            ))
    
        n = int(self._ctrl.nReadDataLen)

        # raw view (нулевая копия)
        raw_view = np.ctypeslib.as_array(self._ch1_buf)[:n]
        # raw_view = np.ctypeslib.as_array(self._ch1_buf)[:n].astype(np.uint16, copy=False)

        # t axis — кэшированный (при copy=True отдаём копию, иначе общий массив)
        t_out = self._t_cache[:n].copy() if copy else self._t_cache[:n]

        volts_out : Optional[np.ndarray] = None
        raw_ret : Optional[np.ndarray] = None

        if return_mode in {"raw", "both"}:
            raw_ret = raw_view.copy() if copy else raw_view

        if return_mode in {"volts", "both"}:
            volts_view = self._raw_to_volts(raw_view, p)
            volts_out = volts_view.copy() if copy else volts_view

        return AScanFrame(
            point=point,
            fs_hz=self.fs_hz,
            dt_s=self.dt_s,
            t_s=t_out,
            triggered=triggered,
            volts=volts_out,
            raw_u16=raw_ret,
        )
    
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

    def _find_device(self) -> int:
        for idx in range(self.max_devices):
            if self.lib.dsoHTDeviceConnect(WORD(idx)):
                return idx
        raise HantekError("Device not found (dsoHTDeviceConnect==0 for all indices).")

    def _ensure_buffers(self, n : int) -> None:
        if self._buf_len >= n and self._ch1_buf is not None:
            return
        self._buf_len = n
        self._ch1_buf = (WORD * n)()
        self._ch2_buf = (WORD * n)()
        self._ch3_buf = (WORD * n)()
        self._ch4_buf = (WORD * n)()

    def _fill_structs(self, p : ScanParams) -> None:
        # RelayControl: включаем только CH1 (остальные выключены)
        for ch in range(MAX_CH_NUM):
            enabled = 1 if ch == int(Channel.CH1) else 0
            self._relay.bCHEnable[ch] = enabled
            self._relay.nCHVoltDIV[ch] = WORD(p.volt_div_idx)
            self._relay.nCHCoupling[ch] = WORD(int(p.coupling))
            self._relay.bCHBWLimit[ch] = 1 if p.bw_limit else 0

        self._relay.nTrigSource = WORD(int(p.trig_source))
        self._relay.bTrigFilt = 0
        self._relay.nALT = 0

        # ControlData
        ch_mask = 0x01  # CH1
        self._ctrl.nCHSet = WORD(ch_mask)
        self._ctrl.nTimeDIV = WORD(p.time_div_idx)
        self._ctrl.nTriggerSource = WORD(int(p.trig_source))
        self._ctrl.nHTriggerPos = WORD(p.pretrigger_percent)
        self._ctrl.nVTriggerPos = WORD(p.trig_level_pos)
        self._ctrl.nTriggerSlope = WORD(int(p.trig_slope))
        self._ctrl.nBufferLen = UINT(p.read_len)
        self._ctrl.nReadDataLen = UINT(p.read_len)
        self._ctrl.nAlreadyReadLen = UINT(0)
        self._ctrl.nALT = 0
        self._ctrl.nETSOpen = 0
        self._ctrl.nDriverCode = 0
        self._ctrl.nLastAddress = 0
        self._ctrl.nFPGAVersion = 0

    @staticmethod
    def _build_time_axis(*, n : int, dt : float, pretrigger_percent : int) -> np.ndarray:
        trig_idx = int(round(n * (pretrigger_percent / 100.0)))
        return (np.arange(n, dtype=np.float64) - trig_idx) * dt

    @staticmethod
    def _raw_to_volts(raw_u16 : np.ndarray, p : ScanParams) -> np.ndarray:
        """
        Перевод raw -> volts
        Здесь сосредоточено место, которое вы будете “калибровать” под свой тракт:
          - деление на 32 (кванта/деление)
          - смещение через lever_pos
          - умножение на V/div
        """
        vdiv = VOLT_DIV_V[p.volt_div_idx]
        return (raw_u16.astype(np.float64) - (MAX_DATA - p.lever_pos)) / 32.0 * vdiv


def main() -> None:
    test()


def test() -> None:
    dll_dir = Path(__file__).resolve().parent / "Hantek_Software" / "Hantek_SDK" / "Dll" / "x64"

    params = ScanParams(
        time_div_idx=9,
        read_len=0x4000,
        volt_div_idx=5,
        pretrigger_percent=50,
        trig_level_pos=128,
        # остальное по умолчанию
    )

    with HantekHardDll(dll_dir) as scope:
        scope.configure(params)  # один раз перед сканом

        for (x, y) in grid_points:
            move_probe_to(x, y)   # ваша программа
            frame = scope.capture(point=(x, y), return_mode="volts", copy=False)

            # frame.point, frame.t_s, frame.volts
            # Важно: copy=False => frame.volts перезапишется на следующем capture().
            # Если нужно хранить — используйте copy=True или сразу сохраняйте/обрабатывайте.


if __name__ == "__main__":
    raise SystemExit(main())