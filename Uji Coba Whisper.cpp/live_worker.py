import threading
import queue
import sounddevice as sd
import numpy as np

# Konfigurasi rekaman
SAMPLE_RATE = 16000  # 16kHz, standar untuk Whisper
CHUNK_DURATION_SEC = 4  # Merekam per 4 detik
BUFFER_SIZE = CHUNK_DURATION_SEC * SAMPLE_RATE

class LiveWorker(threading.Thread):
    """
    Thread untuk merekam audio secara live dan mengirimkannya 
    ke WhisperEngine untuk transkripsi.
    """
    def __init__(self, whisper_engine, model_name, ui_queue):
        super().__init__()
        self.engine = whisper_engine
        self.model_name = model_name
        self.ui_queue = ui_queue  # Antrian untuk kirim hasil ke UI
        self._stop_event = threading.Event()
        
        # Cek apakah model ada sebelum memulai
        try:
            self.engine._load_model(self.model_name)
        except FileNotFoundError:
            self.ui_queue.put(f"ERROR: Model '{self.model_name}' tidak ditemukan.")
            self._stop_event.set() # Langsung stop jika model tidak ada

    def run(self):
        """
        Loop utama thread: rekam, transkripsi, kirim hasil.
        """
        print(f"LiveWorker (model: {self.model_name}) dimulai...")
        try:
            # Buka stream audio
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=BUFFER_SIZE
            ) as stream:
                while not self._stop_event.is_set():
                    # 1. Rekam satu chunk (memblokir selama CHUNK_DURATION_SEC)
                    audio_chunk, overflowed = stream.read(BUFFER_SIZE)
                    
                    if overflowed:
                        print("Peringatan: Audio buffer overflow!")
                    
                    # Konversi ke array 1D
                    audio_numpy = audio_chunk.flatten()

                    # 2. Transkripsi
                    # Ini adalah operasi yang "berat" (CPU-bound)
                    text, delay = self.engine.transcribe_segment(
                        self.model_name, 
                        audio_numpy
                    )
                    
                    # 3. Kirim hasil ke UI
                    if not self._stop_event.is_set():
                        # Kirim sebagai dictionary
                        self.ui_queue.put({
                            "text": text,
                            "delay": delay
                        })
                        
        except sd.PortAudioError as e:
            print(f"Error audio device: {e}")
            self.ui_queue.put(f"ERROR: Tidak bisa membuka mikrofon. {e}")
        except Exception as e:
            print(f"Error di LiveWorker: {e}")
            self.ui_queue.put(f"ERROR: {e}")
            
        print(f"LiveWorker (model: {self.model_name}) berhenti.")

    def stop(self):
        """
        Sinyal untuk menghentikan thread.
        """
        self._stop_event.set()