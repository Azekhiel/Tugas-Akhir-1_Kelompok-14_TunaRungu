import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import queue
import jiwer
import time

from whisper_engine import WhisperEngine
from live_worker import LiveWorker

# Model yang tersedia
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

class WhisperEvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Evaluation Tool")
        self.root.geometry("800x600")

        # Inisialisasi Engine
        self.engine = WhisperEngine()

        # Variabel untuk Mode 1
        self.live_worker_thread = None
        self.live_ui_queue = queue.Queue()
        self.live_segments = []
        self.live_delays = []

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        # Gunakan tema jika ada
        try:
            from ttkthemes import ThemedTk
            if isinstance(self.root, ThemedTk):
                self.root.set_theme("arc")
        except ImportError:
            pass # Jalan tanpa tema

        self.notebook = ttk.Notebook(self.root)
        
        # --- Tab 1: Live Captioning ---
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_live, text='Mode 1: Live Captioning')
        self._setup_tab_live()

        # --- Tab 2: Evaluasi File ---
        self.tab_file = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_file, text='Mode 2: Evaluasi File')
        self._setup_tab_file()

        self.notebook.pack(expand=True, fill='both')

    # --- Setup UI Tab 1: Live ---
    def _setup_tab_live(self):
        # Frame Kontrol
        control_frame = ttk.LabelFrame(self.tab_live, text="Kontrol", padding=10)
        control_frame.pack(fill='x', pady=5)

        ttk.Label(control_frame, text="Pilih Model:").pack(side='left', padx=5)
        self.live_model_var = tk.StringVar(value=AVAILABLE_MODELS[1]) # Default 'base'
        self.live_model_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.live_model_var, 
            values=AVAILABLE_MODELS, 
            state='readonly'
        )
        self.live_model_combo.pack(side='left', padx=5)

        self.live_start_btn = ttk.Button(
            control_frame, 
            text="Start", 
            command=self.start_live_caption
        )
        self.live_start_btn.pack(side='left', padx=5)

        self.live_stop_btn = ttk.Button(
            control_frame, 
            text="Stop & Evaluasi", 
            command=self.stop_live_caption, 
            state='disabled'
        )
        self.live_stop_btn.pack(side='left', padx=5)
        
        # Frame Teks Referensi
        ref_frame = ttk.LabelFrame(self.tab_live, text="Teks Referensi (Ground Truth)", padding=10)
        ref_frame.pack(fill='x', pady=5)
        self.live_ref_text = ScrolledText(ref_frame, height=5, wrap=tk.WORD)
        self.live_ref_text.pack(fill='x', expand=True)

        # Frame Hasil
        result_frame = ttk.LabelFrame(self.tab_live, text="Hasil Transkripsi", padding=10)
        result_frame.pack(fill='both', expand=True, pady=5)
        
        self.live_status_label = ttk.Label(result_frame, text="Status: Idle")
        self.live_status_label.pack(anchor='w')
        
        self.live_result_text = ScrolledText(result_frame, height=15, wrap=tk.WORD, state='disabled')
        self.live_result_text.pack(fill='both', expand=True)

    # --- Setup UI Tab 2: File ---
    def _setup_tab_file(self):
        # Frame Input
        input_frame = ttk.LabelFrame(self.tab_file, text="Input", padding=10)
        input_frame.pack(fill='x', pady=5)

        self.file_audio_btn = ttk.Button(input_frame, text="1. Upload File Audio (.wav, .mp3)", command=self.load_audio_file)
        self.file_audio_btn.pack(fill='x', pady=2)
        self.file_audio_label = ttk.Label(input_frame, text="File: (Belum ada)")
        self.file_audio_label.pack(anchor='w', padx=5)
        self.audio_file_path = ""

        ttk.Label(input_frame, text="2. Masukkan Teks Referensi (Ground Truth):").pack(anchor='w', pady=(10,0))
        self.file_ref_text = ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.file_ref_text.pack(fill='x', expand=True, pady=2)

        # Frame Model
        model_frame = ttk.LabelFrame(self.tab_file, text="3. Pilih Model Evaluasi", padding=10)
        model_frame.pack(fill='x', pady=5)
        
        self.file_model_vars = {}
        for model in AVAILABLE_MODELS:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(model_frame, text=model, variable=var)
            chk.pack(side='left', padx=10)
            self.file_model_vars[model] = var

        # Frame Tombol & Hasil
        action_frame = ttk.Frame(self.tab_file, padding=10)
        action_frame.pack(fill='x')
        
        self.file_eval_btn = ttk.Button(action_frame, text="4. Mulai Evaluasi", command=self.start_file_evaluation)
        self.file_eval_btn.pack()

        self.file_result_text = ScrolledText(action_frame, height=15, wrap=tk.WORD, state='disabled')
        self.file_result_text.pack(fill='both', expand=True, pady=10)

    # --- Logika Aplikasi ---

    # --- Logika Mode 1 (Live) ---
    
    def start_live_caption(self):
        model_name = self.live_model_var.get()
        if self.live_worker_thread:
            messagebox.showwarning("Peringatan", "Sesi live sudah berjalan.")
            return

        # Reset
        self.live_segments = []
        self.live_delays = []
        self.live_result_text.config(state='normal')
        self.live_result_text.delete('1.0', tk.END)
        self.live_result_text.config(state='disabled')
        
        self.live_status_label.config(text=f"Memulai model '{model_name}'...")
        self.root.update_idletasks() # Paksa UI update

        # Mulai thread worker
        self.live_ui_queue = queue.Queue()
        self.live_worker_thread = LiveWorker(
            self.engine, 
            model_name, 
            self.live_ui_queue
        )
        self.live_worker_thread.start()

        # Update UI
        self.live_start_btn.config(state='disabled')
        self.live_stop_btn.config(state='normal')
        self.live_model_combo.config(state='disabled')
        
        # Mulai memantau antrian (queue) dari thread
        self.root.after(100, self.check_live_queue)

    def check_live_queue(self):
        """
        Periksa queue dari worker thread tanpa memblokir UI.
        """
        try:
            while True:
                # Ambil data dari queue
                result = self.live_ui_queue.get_nowait()
                
                if "ERROR:" in str(result):
                    messagebox.showerror("Error Thread", result)
                    self.stop_live_caption(show_eval=False)
                    return

                text = result.get("text", "")
                delay = result.get("delay", 0)

                # Simpan data
                self.live_segments.append(text)
                self.live_delays.append(delay)

                # Update UI
                self.live_result_text.config(state='normal')
                self.live_result_text.insert(tk.END, f"{text} ")
                self.live_result_text.see(tk.END) # Auto-scroll
                self.live_result_text.config(state='disabled')
                
                avg_delay = sum(self.live_delays) / len(self.live_delays)
                self.live_status_label.config(text=f"Status: Merekam... (Avg. Seg. Delay: {avg_delay:.2f} dtk)")

        except queue.Empty:
            # Jika queue kosong, tidak ada masalah. Cek lagi nanti.
            pass
        
        # Jadwalkan cek berikutnya jika thread masih jalan
        if self.live_worker_thread and self.live_worker_thread.is_alive():
            self.root.after(100, self.check_live_queue)

    def stop_live_caption(self, show_eval=True):
        if not self.live_worker_thread:
            return

        # Kirim sinyal stop ke thread
        self.live_worker_thread.stop()
        self.live_worker_thread = None # Hapus referensi

        # Update UI
        self.live_start_btn.config(state='normal')
        self.live_stop_btn.config(state='disabled')
        self.live_model_combo.config(state='normal')
        self.live_status_label.config(text="Status: Idle. Menyiapkan evaluasi...")
        
        if not show_eval:
            self.live_status_label.config(text="Status: Dihentikan.")
            return

        # --- Lakukan Evaluasi Final ---
        reference = self.live_ref_text.get("1.0", tk.END).strip()
        hypothesis = " ".join(self.live_segments).strip()
        
        if not reference or not hypothesis:
            messagebox.showinfo("Evaluasi", "Tidak ada teks referensi atau hasil untuk dievaluasi.")
            self.live_status_label.config(text="Status: Idle.")
            return

        try:
            measures = jiwer.compute_measures(reference, hypothesis)
            wer = measures['wer'] * 100
            mer = measures['mer'] * 100
            wil = measures['wil'] * 100
            
            avg_delay = sum(self.live_delays) / len(self.live_delays) if self.live_delays else 0

            report = f"""
--- EVALUASI FINAL (MODE 1) ---
Model: {self.live_model_var.get()}
Rata-rata Delay Segmen: {avg_delay:.3f} detik
Total Segmen: {len(self.live_segments)}

Word Error Rate (WER): {wer:.2f} %
MER (Match Error Rate): {mer:.2f} %
WIL (Word Info. Lost): {wil:.2f} %

--- Detail ---
Substitution (S): {measures['substitutions']}
Deletion (D): {measures['deletions']}
Insertion (I): {measures['insertions']}
Total Kata Referensi: {measures['truth']}
Total Kata Hasil: {measures['hypothesis']}

--- Teks Referensi ---
{reference}

--- Teks Hasil ---
{hypothesis}
"""
            # Tampilkan di jendela baru
            self.show_report_window(report)
            self.live_status_label.config(text=f"Status: Selesai. Final WER: {wer:.2f} %")

        except Exception as e:
            messagebox.showerror("Error Evaluasi", f"Gagal menghitung WER: {e}")
            self.live_status_label.config(text="Status: Gagal evaluasi.")

    # --- Logika Mode 2 (File) ---

    def load_audio_file(self):
        path = filedialog.askopenfilename(
            title="Pilih File Audio",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if path:
            self.audio_file_path = path
            self.file_audio_label.config(text=f"File: {os.path.basename(path)}")

    def start_file_evaluation(self):
        # Validasi Input
        if not self.audio_file_path:
            messagebox.showerror("Error", "Silakan upload file audio terlebih dahulu.")
            return
            
        reference = self.file_ref_text.get("1.0", tk.END).strip()
        if not reference:
            messagebox.showerror("Error", "Silakan masukkan teks referensi.")
            return
            
        models_to_run = [model for model, var in self.file_model_vars.items() if var.get()]
        if not models_to_run:
            messagebox.showerror("Error", "Silakan pilih minimal satu model untuk evaluasi.")
            return
            
        # Tampilkan status
        self.file_result_text.config(state='normal')
        self.file_result_text.delete('1.0', tk.END)
        self.file_result_text.insert('1.0', "Memulai evaluasi...\n\n")
        self.root.update_idletasks()
        
        # Mulai evaluasi
        report_lines = []
        report_lines.append(f"--- LAPORAN EVALUASI FILE ---")
        report_lines.append(f"File: {os.path.basename(self.audio_file_path)}")
        report_lines.append(f"Teks Referensi: {reference[:100]}...\n")
        report_lines.append("=" * 30)
        
        for model_name in models_to_run:
            self.file_result_text.insert(tk.END, f"\nMemproses dengan model '{model_name}'...\n")
            self.root.update_idletasks()
            
            try:
                # Panggil Engine
                hypothesis, process_time = self.engine.transcribe_file(
                    model_name, 
                    self.audio_file_path
                )
                
                if "Error:" in hypothesis:
                    report_lines.append(f"\nMODEL: {model_name}")
                    report_lines.append(f"  STATUS: GAGAL ({hypothesis})")
                    continue

                # Hitung WER
                measures = jiwer.compute_measures(reference, hypothesis)
                wer = measures['wer'] * 100
                
                # Tambah ke laporan
                report_lines.append(f"\nMODEL: {model_name}")
                report_lines.append(f"  Waktu Proses: {process_time:.2f} detik")
                report_lines.append(f"  WER: {wer:.2f} %")
                report_lines.append(f"  S/D/I: {measures['substitutions']}/{measures['deletions']}/{measures['insertions']}")
                report_lines.append(f"  Teks Hasil: {hypothesis}")
                
                # Update UI
                self.file_result_text.insert(tk.END, f"  -> Selesai! (WER: {wer:.2f} %)\n")

            except Exception as e:
                self.file_result_text.insert(tk.END, f"  -> GAGAL: {e}\n")
                report_lines.append(f"\nMODEL: {model_name}")
                report_lines.append(f"  STATUS: GAGAL ({e})")

        # Tampilkan laporan lengkap
        self.file_result_text.config(state='normal')
        self.file_result_text.delete('1.0', tk.END)
        self.file_result_text.insert('1.0', "\n".join(report_lines))
        self.file_result_text.config(state='disabled')
        
    def show_report_window(self, report_text):
        """
        Menampilkan laporan di jendela pop-up baru.
        """
        report_win = tk.Toplevel(self.root)
        report_win.title("Laporan Evaluasi")
        report_win.geometry("600x400")
        
        text_area = ScrolledText(report_win, wrap=tk.WORD, width=80, height=25)
        text_area.pack(padx=10, pady=10, fill="both", expand=True)
        text_area.insert(tk.END, report_text)
        text_area.config(state="disabled")
        
        close_btn = ttk.Button(report_win, text="Tutup", command=report_win.destroy)
        close_btn.pack(pady=5)
        report_win.transient(self.root) 
        report_win.grab_set()


# --- Main ---
if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()
        
    app = WhisperEvalApp(root)
    root.mainloop()