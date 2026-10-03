"""
Генератор простой мелодии
Создаёт WAV-файл из последовательности нот, используя синусоидальные волны.
"""
 
import numpy as np
import wave
 
# Словарь: нота -> частота в герцах (нотация научной высоты тона)
NOTE_FREQUENCIES = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    'F4': 349.23,
    'G4': 392.00,
    'A4': 440.00,
    'B4': 493.88,
    'C5': 523.25,
}
 
SAMPLE_RATE = 44100  # стандартная частота дискретизации (сэмплов в секунду)
 
 
def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE, amplitude=0.3):
    """
    Генерирует один тон заданной частоты и длительности.
    Возвращает numpy-массив с сэмплами звуковой волны.
    """
    # Массив временных точек: от 0 до duration, с шагом 1/sample_rate
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
 
    # Формула синусоиды: A * sin(2*pi*f*t)
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
 
    # Плавное затухание в конце, чтобы не было щелчка на стыке нот
    fade_samples = int(sample_rate * 0.01)  # 10 мс на затухание
    fade_out = np.linspace(1, 0, fade_samples)
    wave_data[-fade_samples:] *= fade_out
 
    return wave_data
 
 
def generate_melody(notes, note_duration=0.4):
    """
    Принимает список названий нот и собирает из них мелодию.
    Возвращает единый numpy-массив со всей мелодией.
    """
    melody = np.array([], dtype=np.float32)
 
    for note in notes:
        frequency = NOTE_FREQUENCIES[note]
        tone = generate_tone(frequency, note_duration)
        melody = np.concatenate([melody, tone])
 
    return melody
 
 
def save_wav(filename, audio_data, sample_rate=SAMPLE_RATE):
    """
    Сохраняет numpy-массив с аудио в виде WAV-файла.
    """
    # Переводим значения из диапазона [-1, 1] в 16-битный целочисленный формат
    audio_int16 = np.int16(audio_data * 32767)
 
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)          # моно
        wav_file.setsampwidth(2)          # 2 байта = 16 бит
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
 
 
if __name__ == "__main__":
    # Простая мелодия — гамма вверх и вниз
    melody_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
                     'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
 
    print("Генерируем мелодию...")
    melody = generate_melody(melody_notes)
 
    output_file = "melody.wav"
    save_wav(output_file, melody)
 
    print(f"Готово! Мелодия сохранена в файл: {output_file}")
    print(f"Длительность: {len(melody) / SAMPLE_RATE:.2f} секунд")