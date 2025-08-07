import os
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 1  # in seconds

def record_sample(filename):
    print(f"Recording: {filename}")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("Saved:", filename)

def record_batch(folder, label, count):
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        filename = os.path.join(folder, f"{label}_{i+1}.wav")
        input(f"\nPress ENTER to record sample {i+1}/{count}")
        record_sample(filename)

if __name__ == "__main__":
    record_batch("voiceData/wakeWord", "jarvis", 30)
    record_batch("voiceData/randomNoise", "random", 30)
