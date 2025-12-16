import os
from whisper_cpp_python import Whisper
import numpy as np
import soundfile as sf
import time


MODEL_DIR = "./models"

WHISPER_PARAMS = {
    'n_threads': os.cpu_count() or 4,
    'use_gpu': False,  # Set True jika pake NVIDIA GPU & CUDA
    'gpu_device': 0,
}

class WhisperEngine:
    """
    Class wrapper untuk memuat dan menjalankan model whisper.cpp.
    """
    def __init__(self):
        self.models = {}  # Cache untuk menyimpan model yang sudah di-load
        print("WhisperEngine siap.")

    def _load_model(self, model_name="base"):
        """
        Memuat model ke memori jika belum ada.
        """
        if model_name not in self.models:
            print(f"Memuat model '{model_name}'... (Mungkin perlu beberapa detik)")
            model_path = os.path.join(MODEL_DIR, f"ggml-{model_name}.bin")
            
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            self.models[model_name] = Whisper(
                model_path=model_path,
                whisper_params=WHISPER_PARAMS
            )
            print(f"Model '{model_name}' berhasil dimuat.")
        
        return self.models[model_name]

    def transcribe_file(self, model_name, audio_file_path):
        """
        Mentranskripsi seluruh file audio. (Untuk Mode 2)
        
        Mengembalikan: (teks_hasil, waktu_proses_detik)
        """
        try:
            model = self._load_model(model_name)
            
            # Baca file audio
            audio_data, samplerate = sf.read(audio_file_path, dtype='float32')
            
            # Konversi ke 16kHz jika perlu (Whisper internal)
            # whisper-cpp-python menangani ini, tapi pastikan audio mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1) # Konversi ke mono

            print(f"Mulai transkripsi file dengan model '{model_name}'...")
            t_start = time.time()
            
            # Transkripsi
            result = model.transcribe(audio_data, language='id') # Paksa Bahasa Indonesia
            
            t_end = time.time()
            
            full_text = " ".join([seg['text'] for seg in result['segments']])
            process_time = t_end - t_start
            
            print(f"Transkripsi '{model_name}' selesai dalam {process_time:.2f} detik.")
            return full_text, process_time

        except Exception as e:
            print(f"Error saat transkripsi file: {e}")
            return f"Error: {e}", 0

    def transcribe_segment(self, model_name, audio_numpy_array):
        """
        Mentranskripsi potongan audio (numpy array). (Untuk Mode 1)
        
        Mengembalikan: (teks_hasil, waktu_proses_detik)
        """
        try:
            model = self._load_model(model_name)
            
            t_start = time.time()
            
            # Transkripsi
            result = model.transcribe(audio_numpy_array, language='id')
            
            t_end = time.time()
            
            full_text = " ".join([seg['text'] for seg in result['segments']])
            process_time = t_end - t_start
            
            return full_text.strip(), process_time
        
        except Exception as e:
            print(f"Error saat transkripsi segmen: {e}")
            return f"Error: {e}", 0