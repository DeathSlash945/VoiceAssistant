from core.audioStream import AudioStream
from AudioDetection.AudioProcessor import AudioProcessor

class VoiceAssistant:
    def __init__(self):
        self.processor = AudioProcessor()
        self.stream = AudioStream(callback=self.processor.handle_audio)

    def run(self):
        print("Voice Assistant Initialized. Press Enter to start/stop listening. 'q' to quit.")
        listening = False
        try:
            while True:
                cmd = input().strip().lower()
                if cmd == 'q':
                    if listening:
                        self.stream.stop()
                    break
                elif cmd == '':
                    if listening:
                        self.stream.stop()
                        print("Stopped listening.")
                        listening = False
                    else:
                        self.stream.start()
                        print("Listening... Press Enter to stop.")
                        listening = True
        except KeyboardInterrupt:
            if listening:
                self.stream.stop()
