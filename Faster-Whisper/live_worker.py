import threading
import queue
import sounddevice as sd
import numpy as np

# --- KONFIGURASI HARDWARE ORANGE PI ---
# Kita gunakan 44100Hz agar hardware senang (tidak error hw params)
# Nanti kita downsample (kecilkan) ke 16000Hz untuk Whisper
HARDWARE_RATE = 44100  
WHISPER_RATE = 16000
CHANNELS_HW = 2  # WAJIB 2 (Stereo) agar mic Orange Pi mau jalan

# Hitung Blocksize: Seberapa sering callback dipanggil (misal tiap 0.5 detik)
BLOCK_DURATION = 1 # detik
BLOCK_SIZE = int(HARDWARE_RATE * BLOCK_DURATION)

class LiveWorker(threading.Thread):
    def __init__(self, whisper_engine, model_name, ui_queue):
        super().__init__()
        self.engine = whisper_engine
        self.model_name = model_name
        self.ui_queue = ui_queue
        self._stop_event = threading.Event()
        
        # Queue audio
        self.audio_queue = queue.Queue()

        # Load model
        try:
            self.engine._load_model(self.model_name)
        except Exception as e:
            self.ui_queue.put(f"ERROR: {e}")
            self._stop_event.set()

    def run(self):
        print(f"LiveWorker dimulai dengan model: {self.model_name}")
        
        # 1. Jalankan Perekaman (akan berjalan di background via callback)
        # Kita buat thread terpisah untuk menjaga stream tetap hidup
        record_thread = threading.Thread(target=self._start_audio_stream)
        record_thread.start()

        # 2. Loop Utama: Fokus Transkripsi
        while not self._stop_event.is_set():
            try:
                # Ambil data dari queue
                # Data ini sudah Mono (karena sudah dipotong di callback)
                audio_hw = self.audio_queue.get(timeout=1)
                
                # --- PROSES RESAMPLING SEDERHANA (44.1k -> 16k) ---
                # Whisper butuh 16000Hz. Karena kita rekam di 44100Hz,
                # kita perlu sesuaikan. Cara termudah tanpa library berat:
                # Kita "loncat" ambil datanya (Decimation).
                # Rasio 44100 / 16000 ~= 2.75. Kita bulatkan ambil tiap data ke-3.
                # (Ini kasar, tapi cukup untuk speech recognition & hemat CPU)
                audio_16k = audio_hw[::3] 

                # Transkripsi
                text, delay = self.engine.transcribe_segment(
                    self.model_name, 
                    audio_16k
                )
                
                if text.strip():
                    self.ui_queue.put({
                        "text": text,
                        "delay": delay
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error Transkripsi: {e}")

        record_thread.join()
        print("LiveWorker berhenti.")

    def _audio_callback(self, indata, frames, time, status):
        """
        Fungsi ini dipanggil otomatis oleh SoundDevice setiap kali buffer penuh.
        Di sini kita lakukan trik STEREO -> MONO.
        """
        if status:
            print(f"Status Audio: {status}")

        # --- FIX CRASH ORANGE PI DI SINI ---
        # Data masuk bentuknya 2 Dimensi (Stereo): [[Kiri, Kanan], [Kiri, Kanan], ...]
        # Kita ambil kolom 0 saja (Kiri) agar jadi Mono
        mono_audio = indata[:, 0] 
        
        # Masukkan ke antrian untuk diproses thread utama
        # Kita copy() agar aman thread-safe
        self.audio_queue.put(mono_audio.copy())

    def _start_audio_stream(self):
        """
        Membuka stream audio dengan mode Callback
        """
        try:
            # Perhatikan parameternya:
            # samplerate=44100 (Biar hardware tidak error)
            # channels=2 (Biar hardware tidak error)
            # callback=self._audio_callback (Fungsi yang dipanggil otomatis)
            with sd.InputStream(
                samplerate=HARDWARE_RATE, 
                channels=CHANNELS_HW, 
                dtype='float32', 
                blocksize=BLOCK_SIZE, 
                callback=self._audio_callback
            ):
                # Stream akan hidup selama kita stuck di loop ini
                while not self._stop_event.is_set():
                    self._stop_event.wait(1) # Tunggu 1 detik, cek lagi
                    
        except Exception as e:
            self.ui_queue.put(f"ERROR Mic: {e}")
            print(f"Error Stream: {e}")

    def stop(self):
        self._stop_event.set()
