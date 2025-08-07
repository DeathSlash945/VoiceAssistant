import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from core.config import SAMPLE_RATE as sampRate, LOW_CUTOFF_HZ as lowCut, HIGH_CUTOFF_HZ as highCut

class AudioFilter:
    def __init__(self, sample_rate=sampRate):
        self.sample_rate = sample_rate
        self.sos, self.zi = self.design_bandpass()

    def design_bandpass(self):
        nyquist = self.sample_rate / 2.0
        low = lowCut / nyquist
        high = highCut / nyquist

        if not (0 < low < high < 1):
            raise ValueError("Invalid normalized frequencies")

        sos = butter(N=4, Wn=[low, high], btype='band', output='sos')
        zi = sosfilt_zi(sos)
        return sos, zi

    def apply(self, audio_chunk):
        return sosfilt(self.sos, audio_chunk)
