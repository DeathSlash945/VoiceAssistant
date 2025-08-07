from collections import deque
from AudioDetection.Filter import AudioFilter
from AudioDetection.VoiceDetection import VoiceDetector
from AudioDetection.Extraction import FeatureExtractor

class AudioProcessor:
    def __init__(self):
        self.filter = AudioFilter()
        self.vad = VoiceDetector()
        self.extractor = FeatureExtractor()
        self.audio_buffer = deque(maxlen=16000 * 3)

    def handle_audio(self, chunk):
        self.audio_buffer.extend(chunk)
        normalized = self.normalize(chunk)
        filtered = self.filter.apply(normalized)

        is_speech, energy = self.vad.detect(filtered)
        if is_speech:
            print(f"Speech detected, energy: {energy:.4f}")
            features = self.extractor.extract(filtered)
            print(f"MFCC shape: {features.shape}")

    #to bring it in -1 to 1 range
    def normalize(self, chunk):
        max_val = max(abs(chunk.max()), abs(chunk.min()))
        return chunk / max_val if max_val else chunk
