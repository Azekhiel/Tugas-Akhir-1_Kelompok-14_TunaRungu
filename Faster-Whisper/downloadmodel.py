import os
from faster_whisper import WhisperModel

# Tentukan folder tujuan (Relatif terhadap file script ini)
# Ini akan membuat folder "models" di dalam folder proyek Anda
CUSTOM_MODEL_DIR = "./models"

# Daftar model yang ingin didownload
MODELS_TO_DOWNLOAD = ["deepdml/faster-whisper-large-v3-turbo-ct2"]

def download_models():
    print(f"📂 Folder penyimpanan target: {os.path.abspath(CUSTOM_MODEL_DIR)}")
    
    # Buat folder jika belum ada
    if not os.path.exists(CUSTOM_MODEL_DIR):
        os.makedirs(CUSTOM_MODEL_DIR)

    for model_name in MODELS_TO_DOWNLOAD:
        print(f"\n⬇️  Sedang memproses model: '{model_name}'...")
        try:
            # download_root=... adalah kuncinya!
            model = WhisperModel(
                model_name, 
                device="cpu", 
                compute_type="int8", 
                download_root=CUSTOM_MODEL_DIR 
            )
            print(f"✅ Sukses! Model '{model_name}' tersimpan.")
            del model # Bersihkan RAM
            
        except Exception as e:
            print(f"❌ Gagal download '{model_name}': {e}")

    print("\n🎉 Selesai! Cek folder 'models' Anda sekarang.")

if __name__ == "__main__":
    download_models()