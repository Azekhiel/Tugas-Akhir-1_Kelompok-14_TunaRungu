import threading
import queue
import sounddevice as sd
import numpy as np

# Konfigurasi
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 3 # Kurangi jadi 3 detik agar lebih responsif
BUFFER_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_SEC)

class LiveWorker(threading.Thread):
    def __init__(self, whisper_engine, model_name, ui_queue):
        super().__init__()
        self.engine = whisper_engine
        self.model_name = model_name
        self.ui_queue = ui_queue
        self._stop_event = threading.Event()
        
        # Queue khusus untuk menampung audio mentah dari mic
        # agar perekaman tidak terganggu proses transkripsi
        self.audio_queue = queue.Queue()

        # Load model di awal
        try:
            self.engine._load_model(self.model_name)
        except Exception as e:
            self.ui_queue.put(f"ERROR: {e}")
            self._stop_event.set()

    def run(self):
        """
        Menjalankan dua aktivitas paralel:
        1. Thread Perekam (Recording)
        2. Loop Pemroses (Transcribing)
        """
        print(f"LiveWorker dimulai dengan model: {self.model_name}")
        
        # Jalankan perekaman di thread terpisah (Non-blocking)
        record_thread = threading.Thread(target=self._record_loop)
        record_thread.start()

        # Loop utama ini sekarang HANYA fokus transkripsi
        while not self._stop_event.is_set():
            try:
                # Ambil audio dari antrian (tunggu max 1 detik)
                audio_numpy = self.audio_queue.get(timeout=1)
                
                # Transkripsi
                text, delay = self.engine.transcribe_segment(
                    self.model_name, 
                    audio_numpy
                )
                
                # Kirim ke UI hanya jika ada teks
                if text.strip():
                    self.ui_queue.put({
                        "text": text,
                        "delay": delay
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error Transkripsi: {e}")

        # Tunggu thread rekam mati
        record_thread.join()
        print("LiveWorker berhenti.")

    def _record_loop(self):
        """
        Fungsi ini terus menerus merekam tanpa henti.
        Audio dilempar ke queue untuk diproses thread utama.
        """
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BUFFER_SIZE) as stream:
                while not self._stop_event.is_set():
                    # Baca audio chunk
                    audio_chunk, overflowed = stream.read(BUFFER_SIZE)
                    if overflowed:
                        print("Warning: Audio Buffer Overflow (CPU overload)")
                    
                    # Masukkan ke antrian untuk diproses
                    self.audio_queue.put(audio_chunk.flatten())
        except Exception as e:
            self.ui_queue.put(f"ERROR Mic: {e}")

    def stop(self):
        self._stop_event.set()