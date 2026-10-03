"""
Реверберация (Reverb) — классический алгоритм Шрёдера:
несколько гребенчатых фильтров (comb) параллельно
+ пара всепропускающих фильтров (allpass) последовательно.
"""

import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100


# ============================================================
# 1. ГРЕБЕНЧАТЫЙ ФИЛЬТР С ОБРАТНОЙ СВЯЗЬЮ (comb filter)
# ============================================================

def comb_filter(signal, delay_samples, feedback_gain):
    """
    y[n] = x[n] + g * y[n - d]

    В отличие от фленджера, здесь в обратную связь идёт УЖЕ
    ОБРАБОТАННЫЙ сигнал (y), а не исходный (x) — поэтому эхо
    не затухает после одного повтора, а звенит долго.
    """
    n = len(signal)
    output = np.zeros(n)

    for i in range(n):
        delayed = output[i - delay_samples] if i - delay_samples >= 0 else 0.0
        output[i] = signal[i] + feedback_gain * delayed

    return output


# ============================================================
# 2. ВСЕПРОПУСКАЮЩИЙ ФИЛЬТР (allpass filter)
# ============================================================

def allpass_filter(signal, delay_samples, gain=0.5):
    """
    y[n] = -g * x[n] + x[n - d] + g * y[n - d]

    Не меняет громкость частот (отсюда название "всепропускающий"),
    но размазывает эхо во времени, убирая металлический призвук.
    """
    n = len(signal)
    output = np.zeros(n)

    for i in range(n):
        x_delayed = signal[i - delay_samples] if i - delay_samples >= 0 else 0.0
        y_delayed = output[i - delay_samples] if i - delay_samples >= 0 else 0.0
        output[i] = -gain * signal[i] + x_delayed + gain * y_delayed

    return output


# ============================================================
# 3. РЕВЕРБЕРАТОР ШРЁДЕРА — собираем всё вместе
# ============================================================

def schroeder_reverb(signal, sample_rate=SAMPLE_RATE, mix=0.3):
    """
    4 comb-фильтра параллельно (сумма) -> 2 allpass-фильтра последовательно.

    Задержки в мс подобраны так, чтобы не быть кратными друг другу —
    это предотвращает совпадение резонансов гребёнок и звучит естественнее.
    """
    # Задержки и коэффициенты для comb-фильтров (мс -> сэмплы)
    comb_delays_ms = [50.7, 59.1, 72, 235]
    comb_feedback = 0.9

    comb_sum = np.zeros(len(signal))
    for delay_ms in comb_delays_ms:
        delay_samples = int(round(delay_ms * sample_rate / 1000.0))
        comb_sum += comb_filter(signal, delay_samples, comb_feedback)

    comb_sum /= len(comb_delays_ms)  # усредняем, чтобы не было перегруза громкости

    # Два allpass-фильтра последовательно — сглаживают звук
    allpass_delays_ms = [10.0, 5]
    diffused = comb_sum
    for delay_ms in allpass_delays_ms:
        delay_samples = int(round(delay_ms * sample_rate / 1000.0))
        diffused = allpass_filter(diffused, delay_samples, gain=0.5)

    # Финальный микс сухого и реверберированного сигнала
    output = (1 - mix) * signal + mix * diffused
    return output


# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_impulse_response(reverb_func, sample_rate=SAMPLE_RATE, duration=1.5):
    """
    Подаём на вход "импульс" (один щелчок — единица, затем тишина)
    и смотрим, как реверберация "разносит" этот один щелчок во времени.
    Это стандартный способ показать характер любого эффекта/фильтра.
    """
    n = int(sample_rate * duration)
    impulse = np.zeros(n)
    impulse[0] = 1.0  # единственный "щелчок" в самом начале

    response = reverb_func(impulse, sample_rate=sample_rate, mix=1.0)

    t = np.arange(n) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t, response, linewidth=0.7)
    axes[0].set_title("Импульсная характеристика реверберации")
    axes[0].set_xlabel("Время (сек)")
    axes[0].set_ylabel("Амплитуда")

    # Частотный отклик через FFT
    spectrum = np.fft.rfft(response)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    axes[1].plot(freqs, magnitude_db)
    axes[1].set_xscale('log')
    axes[1].set_xlim(20, 20000)
    axes[1].set_title("Частотная характеристика реверберации")
    axes[1].set_xlabel("Частота (Гц)")
    axes[1].set_ylabel("Амплитуда (дБ)")
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig("reverb_response.png", dpi=150)
    plt.show()
    print("График сохранён в reverb_response.png")


if __name__ == "__main__":
    print("Строим импульсную и частотную характеристику реверберации...")
    plot_impulse_response(schroeder_reverb)
    print("Готово.")