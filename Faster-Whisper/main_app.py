import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import queue
import jiwer
import os
import threading
import string

# Pastikan file ini ada di folder yang sama
from whisper_engine import WhisperEngine
from live_worker import LiveWorker

# Model yang tersedia di Faster-Whisper
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]

class WhisperEvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Faster-Whisper Evaluation Tool (Fix Jiwer 3.0)")
        self.root.geometry("1000x750")

        # Inisialisasi Engine
        self.engine = WhisperEngine()

        # Variabel untuk Mode 1 (Live)
        self.live_worker_thread = None
        self.live_ui_queue = queue.Queue()
        self.live_segments = []
        self.live_delays = []

        self._setup_ui()

    # --- FUNGSI PENTING: Normalisasi Teks ---
    def normalize_text(self, text):
        """
        Membersihkan teks agar perhitungan WER adil.
        1. Ubah ke huruf kecil.
        2. Hapus tanda baca (titik, koma, tanda tanya, dll).
        3. Hapus spasi berlebih.
        """
        if not text: return ""
        # Lowercase
        text = text.lower()
        # Hapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Hapus spasi ganda
        text = " ".join(text.split())
        return text

    def _setup_ui(self):
        # Setup Tema (Opsional)
        try:
            from ttkthemes import ThemedTk
            if isinstance(self.root, ThemedTk):
                self.root.set_theme("arc")
        except ImportError:
            pass

        self.notebook = ttk.Notebook(self.root)
        
        # Tab 1: Live
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_live, text='Mode 1: Live Captioning')
        self._setup_tab_live()

        # Tab 2: File
        self.tab_file = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_file, text='Mode 2: Evaluasi File')
        self._setup_tab_file()

        self.notebook.pack(expand=True, fill='both')

    def _setup_tab_live(self):
        # Frame Kontrol
        control_frame = ttk.LabelFrame(self.tab_live, text="Kontrol", padding=10)
        control_frame.pack(fill='x', pady=5)

        ttk.Label(control_frame, text="Pilih Model:").pack(side='left', padx=5)
        self.live_model_var = tk.StringVar(value="small") 
        self.live_model_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.live_model_var, 
            values=AVAILABLE_MODELS, 
            state='readonly',
            width=15
        )
        self.live_model_combo.pack(side='left', padx=5)

        self.live_start_btn = ttk.Button(control_frame, text="Start Live", command=self.start_live_caption)
        self.live_start_btn.pack(side='left', padx=5)

        self.live_stop_btn = ttk.Button(control_frame, text="Stop & Evaluasi", command=self.stop_live_caption, state='disabled')
        self.live_stop_btn.pack(side='left', padx=5)
        
        # Frame Referensi
        ref_frame = ttk.LabelFrame(self.tab_live, text="Teks Referensi (Ground Truth) - Opsional", padding=10)
        ref_frame.pack(fill='x', pady=5)
        self.live_ref_text = ScrolledText(ref_frame, height=10, font=("Arial", 14), wrap=tk.WORD)
        self.live_ref_text.pack(fill='x', expand=True)

        # Frame Hasil
        result_frame = ttk.LabelFrame(self.tab_live, text="Hasil Transkripsi Real-time", padding=10)
        result_frame.pack(fill='both', expand=True, pady=5)
        self.live_status_label = ttk.Label(result_frame, text="Status: Idle")
        self.live_status_label.pack(anchor='w')
        self.live_result_text = ScrolledText(result_frame, height=15, font=("Arial", 14), wrap=tk.WORD, state='disabled')
        self.live_result_text.pack(fill='both', expand=True)

    def _setup_tab_file(self):
        input_frame = ttk.LabelFrame(self.tab_file, text="Input", padding=10)
        input_frame.pack(fill='x', pady=5)

        self.file_audio_btn = ttk.Button(input_frame, text="1. Upload File Audio", command=self.load_audio_file)
        self.file_audio_btn.pack(fill='x', pady=2)
        self.file_audio_label = ttk.Label(input_frame, text="File: (Belum ada)")
        self.file_audio_label.pack(anchor='w', padx=5)
        self.audio_file_path = ""

        ttk.Label(input_frame, text="2. Masukkan Teks Referensi (Ground Truth):").pack(anchor='w', pady=(10,0))
        self.file_ref_text = ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.file_ref_text.pack(fill='x', expand=True, pady=2)

        model_frame = ttk.LabelFrame(self.tab_file, text="3. Pilih Model Evaluasi", padding=10)
        model_frame.pack(fill='x', pady=5)
        
        self.file_model_vars = {}
        for model in AVAILABLE_MODELS:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(model_frame, text=model, variable=var)
            chk.pack(side='left', padx=10)
            self.file_model_vars[model] = var

        action_frame = ttk.Frame(self.tab_file, padding=10)
        action_frame.pack(fill='both', expand=True)
        self.file_eval_btn = ttk.Button(action_frame, text="4. Mulai Evaluasi", command=self.start_file_evaluation)
        self.file_eval_btn.pack(pady=5)

        self.file_result_text = ScrolledText(action_frame, height=15, wrap=tk.WORD, state='disabled')
        self.file_result_text.pack(fill='both', expand=True, pady=5)

    # --- LOGIKA LIVE ---
    def start_live_caption(self):
        model_name = self.live_model_var.get()
        if self.live_worker_thread: return

        # Reset UI
        self.live_segments = []
        self.live_delays = []
        self.live_result_text.config(state='normal')
        self.live_result_text.delete('1.0', tk.END)
        self.live_result_text.config(state='disabled')
        
        self.live_status_label.config(text=f"Menyiapkan model '{model_name}'... (Loading Faster-Whisper)")
        self.root.update_idletasks()
        
        # Init Worker di Thread
        self.live_ui_queue = queue.Queue()
        threading.Thread(target=self._init_worker_thread, args=(model_name,), daemon=True).start()

    def _init_worker_thread(self, model_name):
        try:
            self.live_worker_thread = LiveWorker(self.engine, model_name, self.live_ui_queue)
            self.live_worker_thread.start()
            self.root.after(0, self._on_worker_started)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Gagal init worker: {e}"))

    def _on_worker_started(self):
        self.live_start_btn.config(state='disabled')
        self.live_stop_btn.config(state='normal')
        self.live_model_combo.config(state='disabled')
        self.root.after(100, self.check_live_queue)

    def check_live_queue(self):
        try:
            while True:
                # 1. AMBIL DATA (Ini titik krusial)
                # Jika queue kosong, baris ini akan melempar error 'Empty'
                # dan langsung loncat ke 'except queue.Empty'.
                result = self.live_ui_queue.get_nowait()
                
                # 2. CEK ERROR DARI WORKER
                if isinstance(result, str) and "ERROR:" in result:
                    messagebox.showerror("Error Thread", result)
                    self.stop_live_caption(show_eval=False)
                    return

                # 3. AMBIL TEKS
                # (Baris ini AMAN karena ada di dalam blok 'try' dan setelah 'get_nowait')
                text = result.get("text", "").strip()
                delay = result.get("delay", 0)

                # --- FILTERING LOGIC ---
                if not text:
                    continue

                text_lower = text.lower()

                # A. Filter Halusinasi (Bahasa Inggris & Tanda Baca)
                hallucinations = [
                    "thank you", "thanks for watching", "subtitle", 
                    "captioned", "amara.org", "copyright", 
                    "subscribe", "community contributions", "terima kasih sudah menonton",
                ]
                # Cek halusinasi atau cuma tanda baca
                is_hallucination = any(h in text_lower for h in hallucinations)
                if text in [".", "?", "!", ",", "...", "Says:"]:
                    is_hallucination = True
                if is_hallucination:
                    print(f"👻 Halusinasi Dibuang: '{text}'")
                    continue 

                # B. Filter "Terima Kasih" di AWAL SESI (Start)
                # Jika ini kalimat pertama DAN isinya cuma "terima kasih", buang.
                is_first_sentence = (len(self.live_segments) == 0)
                if is_first_sentence and "terima kasih" in text_lower:
                     # print("🚫 'Terima Kasih' di awal sesi dibuang.")
                     continue

                # C. Filter Duplikat (Mencegah Looping)
                # Jika teks SAMA PERSIS dengan yang terakhir muncul, buang.
                if self.live_segments and text == self.live_segments[-1]:
                    # print(f"♻️ Duplikat Dibuang: '{text}'")
                    continue 

                # --- UPDATE UI (Hanya jika lolos semua filter) ---
                self.live_segments.append(text)
                self.live_delays.append(delay)

                self.live_result_text.config(state='normal')
                self.live_result_text.insert(tk.END, f"{text} ")
                self.live_result_text.see(tk.END)
                self.live_result_text.config(state='disabled')
                
                avg_delay = sum(self.live_delays) / len(self.live_delays) if self.live_delays else 0
                self.live_status_label.config(text=f"Status: Merekam... (Latency: {avg_delay:.2f}s)")

        except queue.Empty:
            # Jika queue kosong, biarkan saja (pass), jangan lakukan apa-apa.
            pass
        
        # Jadwalkan pengecekan ulang 100ms lagi
        if self.live_worker_thread and self.live_worker_thread.is_alive():
            self.root.after(100, self.check_live_queue)

    def stop_live_caption(self, show_eval=True):
        if not self.live_worker_thread: return
        
        # Matikan worker
        self.live_worker_thread.stop()
        self.live_worker_thread = None
        
        # Reset Tombol
        self.live_start_btn.config(state='normal')
        self.live_stop_btn.config(state='disabled')
        self.live_model_combo.config(state='normal')
        
        if not show_eval:
            self.live_status_label.config(text="Status: Dihentikan.")
            return

        # --- EVALUASI WER (JIWER 3.0 FIX) ---
        raw_reference = self.live_ref_text.get("1.0", tk.END).strip()
        raw_hypothesis = " ".join(self.live_segments).strip()

        # 1. Normalisasi
        reference = self.normalize_text(raw_reference)
        hypothesis = self.normalize_text(raw_hypothesis)

        if not reference or not hypothesis:
             self.live_status_label.config(text="Selesai. (Data tidak cukup untuk WER)")
             return

        try:
            # 2. Hitung Metrics dengan library baru
            output = jiwer.process_words(reference, hypothesis)
            wer = output.wer * 100
            mer = output.mer * 100
            wil = output.wil * 100
            
            avg_delay = sum(self.live_delays) / len(self.live_delays) if self.live_delays else 0

            # Laporan
            report = f"""
--- LAPORAN LIVE (FASTER-WHISPER) ---
Model: {self.live_model_var.get()}
Latency Rata-rata: {avg_delay:.3f} detik

WER (Word Error Rate): {wer:.2f} %
MER (Match Error Rate): {mer:.2f} %
WIL (Word Info Lost): {wil:.2f} %

--- Detail Kesalahan ---
Substitution (S): {output.substitutions}
Deletion (D): {output.deletions}
Insertion (I): {output.insertions}

--- Perbandingan (Normalized) ---
Ref: {reference}
Hyp: {hypothesis}
"""
            self.show_report_window(report)
            self.live_status_label.config(text=f"Status: Selesai. WER: {wer:.2f} %")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal hitung WER: {e}")

    # --- LOGIKA FILE ---
    def load_audio_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac"), ("All", "*.*")])
        if path:
            self.audio_file_path = path
            self.file_audio_label.config(text=f"File: {os.path.basename(path)}")

    def start_file_evaluation(self):
        raw_reference = self.file_ref_text.get("1.0", tk.END).strip()
        
        if not self.audio_file_path or not raw_reference:
            messagebox.showerror("Error", "File audio dan Teks Referensi wajib diisi.")
            return
        
        models = [m for m, v in self.file_model_vars.items() if v.get()]
        if not models:
            messagebox.showerror("Error", "Pilih minimal satu model.")
            return

        self.file_result_text.config(state='normal')
        self.file_result_text.delete('1.0', tk.END)
        self.file_result_text.insert('1.0', "Memulai evaluasi Faster-Whisper...\n\n")
        
        # Jalankan di Thread
        threading.Thread(target=self._run_file_eval_thread, args=(models, raw_reference)).start()

    def _run_file_eval_thread(self, models_to_run, raw_reference):
        # Normalisasi Referensi
        reference = self.normalize_text(raw_reference)

        report_lines = [f"--- LAPORAN EVALUASI FILE ---", f"File: {os.path.basename(self.audio_file_path)}", "="*40]
        
        for model_name in models_to_run:
            self.root.after(0, lambda m=model_name: self._append_log(f"Memproses '{m}'... "))
            try:
                # Transkripsi
                raw_hypothesis, process_time = self.engine.transcribe_file(model_name, self.audio_file_path)
                
                # Normalisasi Hipotesis
                hypothesis = self.normalize_text(raw_hypothesis)

                if "Error" in raw_hypothesis:
                    report_lines.append(f"\nMODEL: {model_name} -> GAGAL")
                    continue

                # Hitung WER (Jiwer 3.0 Fix)
                output = jiwer.process_words(reference, hypothesis)
                wer = output.wer * 100
                
                log_text = f"Selesai ({process_time:.2f}s). WER: {wer:.2f}%\n"
                self.root.after(0, lambda l=log_text: self._append_log(l))

                report_lines.append(f"\nMODEL: {model_name}")
                report_lines.append(f"  Waktu: {process_time:.2f}s")
                report_lines.append(f"  WER: {wer:.2f}%")
                report_lines.append(f"  S/D/I: {output.substitutions}/{output.deletions}/{output.insertions}")
                report_lines.append(f"  Hasil (Raw): {raw_hypothesis}")

            except Exception as e:
                self.root.after(0, lambda err=e: self._append_log(f"ERROR: {err}\n"))

        final_report = "\n".join(report_lines)
        self.root.after(0, lambda: self._show_final_file_report(final_report))

    def _append_log(self, text):
        self.file_result_text.config(state='normal')
        self.file_result_text.insert(tk.END, text)
        self.file_result_text.see(tk.END)
        self.file_result_text.config(state='disabled')

    def _show_final_file_report(self, text):
        self.show_report_window(text)
        messagebox.showinfo("Selesai", "Evaluasi File Selesai.")

    def show_report_window(self, text):
        win = tk.Toplevel(self.root)
        win.title("Laporan Evaluasi")
        win.geometry("600x500")
        st = ScrolledText(win)
        st.pack(fill='both', expand=True)
        st.insert(tk.END, text)
        st.config(state='disabled')

if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()
        
    app = WhisperEvalApp(root)
    root.mainloop()