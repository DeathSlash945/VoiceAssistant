import numpy as np

class VoiceDetector:
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def detect(self, audio_chunk):
        if len(audio_chunk) == 0:
            return False, 0.0

        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))

        if len(audio_chunk) > 1:
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk))))
            zcr = zero_crossings / (2 * len(audio_chunk))
        else:
            zcr = 0.0

        is_speech = rms_energy > self.threshold and 0.01 < zcr < 0.5
        return is_speech, rms_energy
