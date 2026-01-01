import os
import time
from faster_whisper import WhisperModel

MODEL_MAP = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2"  # <--- Mapping Baru
}

class WhisperEngine:
    def __init__(self):
        self.models = {}
        self.device = "cpu"
        self.compute_type = "int8"
        self.model_path = "./models" 
        print(f"WhisperEngine siap. Cache: {os.path.abspath(self.model_path)}")

    def _load_model(self, model_name="small"):
        # Terjemahkan nama dulu (misal: "turbo" -> "deepdml/...")
        # Jika nama tidak ada di map, pakai nama aslinya
        real_name = MODEL_MAP.get(model_name, model_name)

        if model_name not in self.models:
            print(f"⏳ Memuat model '{model_name}' ({real_name})...")
            try:
                self.models[model_name] = WhisperModel(
                    real_name,  # <--- Pakai nama asli untuk loading
                    device=self.device, 
                    compute_type=self.compute_type,
                    download_root=self.model_path
                )
                print(f"🚀 Model '{model_name}' siap!")
            except Exception as e:
                print(f"❌ Error: {e}")
                raise e
        return self.models[model_name]

    def transcribe_segment(self, model_name, audio_array):
        """
        Fungsi ini dipanggil oleh LiveWorker.
        Input: Numpy Array (float32) dari mic
        Output: (Teks, Waktu Proses)
        """
        try:
            model = self._load_model(model_name)
            
            start_time = time.time()
            
            # --- INTI FASTER-WHISPER ---
            # beam_size=1: Paling cepat (Greedy)
            # vad_filter=True: Fitur HEBAT faster-whisper. 
            # Dia otomatis membuang bagian hening/nafas sebelum transkripsi.
            segments, info = model.transcribe(
                audio_array, 
                beam_size=1, 
                language="id",    # Fokus Bahasa Indonesia
                vad_filter=True,  # Wajib: Hapus audio hening
                vad_parameters=dict(min_silence_duration_ms=500), 
                condition_on_previous_text=False,  # Wajib: Jangan terjebak pengulangan
                no_speech_threshold=0.5,
                repetition_penalty=1.2,
            )
            
            # Gabungkan hasil segmen
            text_result = " ".join([segment.text for segment in segments]).strip()
            
            process_time = time.time() - start_time
            
            return text_result, process_time

        except Exception as e:
            print(f"Error engine: {e}")
            return "", 0