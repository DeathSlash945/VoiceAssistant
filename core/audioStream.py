import sounddevice as sd
from core.config import SAMPLE_RATE as sampRate, CHUNK_SIZE as chunkSize

class AudioStream:
    def __init__(self, callback):
        self.stream = None
        self.callback = callback

    def start(self):
        self.stream = sd.InputStream(callback=self._internal_callback,
                                     channels=1,
                                     samplerate=sampRate,
                                     blocksize=chunkSize,
                                     dtype='float32')
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _internal_callback(self, indata, frames, time_info, status):
        if status:
            print("Stream warning:", status)
        audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
        self.callback(audio_chunk)
