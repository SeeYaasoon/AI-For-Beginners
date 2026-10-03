"""
Фленджер (Flanger) — эффект на основе переменной задержки
и построение его частотной характеристики (АЧХ).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

SAMPLE_RATE = 44100


# ============================================================
# 1. САМ ЭФФЕКТ ФЛЕНДЖЕРА
# ============================================================

def flanger(signal, sample_rate=SAMPLE_RATE,
            rate_hz=0.3, depth_ms=3.0, base_delay_ms=3.0,
            mix=0.5, feedback=0.2):
    """
    Применяет эффект фленджера к сигналу.

    rate_hz     — скорость качания LFO (сколько раз в секунду "плывёт" задержка)
    depth_ms    — глубина модуляции задержки в миллисекундах
    base_delay_ms — базовая (средняя) задержка в миллисекундах
    mix         — доля обработанного сигнала в итоговом миксе (0..1)
    feedback    — доля сигнала, возвращаемая обратно на вход (усиливает резонансы)
    """
    n = len(signal)
    output = np.zeros(n)

    # LFO (Low Frequency Oscillator) — синусоида, управляющая задержкой во времени
    t = np.arange(n) / sample_rate
    lfo = np.sin(2 * np.pi * rate_hz * t)

    # Задержка в сэмплах, "плывущая" от (base-depth) до (base+depth)
    delay_samples = (base_delay_ms + depth_ms * lfo) * sample_rate / 1000.0

    # Буфер для feedback — храним уже обработанный сигнал
    delayed_signal = np.zeros(n)

    for i in range(n):
        d = delay_samples[i]
        idx = i - d  # дробный индекс в прошлое

        if idx < 0:
            sample = 0.0
        else:
            # Линейная интерполяция между двумя соседними сэмплами,
            # т.к. задержка дробная (не целое число сэмплов)
            idx_floor = int(np.floor(idx))
            frac = idx - idx_floor
            s0 = signal[idx_floor] + feedback * delayed_signal[idx_floor]
            if idx_floor + 1 < n:
                s1 = signal[idx_floor + 1] + feedback * delayed_signal[idx_floor + 1]
            else:
                s1 = s0
            sample = s0 * (1 - frac) + s1 * frac

        delayed_signal[i] = sample
        output[i] = (1 - mix) * signal[i] + mix * sample

    return output

