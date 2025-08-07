import numpy as np
import librosa
from core.config import SAMPLE_RATE as sampRate, LOW_CUTOFF_HZ as lowCut, HIGH_CUTOFF_HZ as highCut

class FeatureExtractor:
    def __init__(self, sample_rate=sampRate):
        self.sample_rate = sample_rate

    def extract(self, audio_chunk):
        try:
            min_len = 512
            if len(audio_chunk) < min_len:
                audio_chunk = np.pad(audio_chunk, (0, min_len - len(audio_chunk)), mode='constant')

            mfcc = librosa.feature.mfcc(
                y=audio_chunk,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=256,
                n_mels=26,
                fmin=lowCut,
                fmax=highCut
            )

            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            combined = np.vstack([mfcc, delta, delta2])

            if combined.shape[1] == 0:
                return np.zeros((1, 39))

            return combined.T

        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return np.zeros((1, 39))
