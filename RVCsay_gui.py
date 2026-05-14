#!/usr/bin/env python3
"""
RVCsay GUI - Text to RVC Voice
Simplified, high-quality RVC inference interface for Windows & Linux.
Requires: ffmpeg, rvc-python, edge-tts, sounddevice, soundfile, numpy
"""

import os
import sys
import glob
import time
import platform
import subprocess
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk

# -- Third-party dependencies --
try:
    import torch
    import sounddevice as sd
    import soundfile as sf
    from rvc_python.infer import RVCInference
except ImportError as e:
    print(f"Missing dependency: {e}. Please run: pip install rvc-python edge-tts sounddevice soundfile numpy")

# -- PyTorch Compatibility Fix (2.6+) --
try:
    import torch
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)
    torch.load = _safe_load
except:
    pass

# ══════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ══════════════════════════════════════════════
APP_NAME      = "RVCsay"
BASE_DIR      = os.path.join(os.path.expanduser("~"), "rvc-voice")
AUDIO_DIR     = os.path.join(BASE_DIR, "audio")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MAX_KEEP_FILES = 5
DEFAULT_LANG  = "th"

# UI Constants
COLOR_BG      = "#0d0d0d"
COLOR_BG_SEC  = "#161616"
COLOR_BG_TER  = "#1e1e1e"
COLOR_ACCENT  = "#e0ff4f"
COLOR_DIM     = "#333333"
COLOR_TEXT    = "#e8e8e8"
COLOR_SUBTEXT = "#666666"
COLOR_RED     = "#ff4f4f"
COLOR_GREEN   = "#4fff91"
COLOR_ORANGE  = "#ffaa4f"

LANGUAGES = {
    "Thai (th)": "th",
    "English (en)": "en",
    "Japanese (ja)": "ja",
    "Chinese (zh)": "zh",
    "Korean (ko)": "ko",
    "French (fr)": "fr",
}

# ══════════════════════════════════════════════
#  ENVIRONMENT SETUP
# ══════════════════════════════════════════════
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Startup info for subprocesses (hiding console on Windows)
SUBPROCESS_STARTUPINFO = None
if IS_WINDOWS:
    SUBPROCESS_STARTUPINFO = subprocess.STARTUPINFO()
    SUBPROCESS_STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    SUBPROCESS_STARTUPINFO.wShowWindow = subprocess.SW_HIDE

def find_ffmpeg() -> str:
    """Finds ffmpeg executable in PATH or common Windows locations."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if IS_WINDOWS:
        common_paths = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p
    return "ffmpeg"

FFMPEG_PATH = find_ffmpeg()

# ══════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════

def cleanup_old_audio(keep=MAX_KEEP_FILES):
    """Keeps the audio directory clean by removing old files."""
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")), key=os.path.getmtime)
    while len(files) >= keep:
        try:
            os.remove(files.pop(0))
        except:
            pass

def play_audio(wav_path):
    """Plays audio using the best available method for the platform."""
    if IS_WINDOWS:
        try:
            data, sr = sf.read(wav_path)
            sd.play(data, sr)
            sd.wait()
            return
        except:
            pass
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            return
        except:
            pass
    elif IS_LINUX:
        try:
            r = subprocess.run(["paplay", wav_path], capture_output=True, timeout=30)
            if r.returncode == 0: return
        except:
            pass
        try:
            subprocess.run(["aplay", "-q", wav_path], timeout=30)
            return
        except:
            pass
    # Fallback
    try:
        data, sr = sf.read(wav_path)
        sd.play(data, sr)
        sd.wait()
    except:
        pass

# ══════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ══════════════════════════════════════════════

class RVCsayApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.configure(bg=COLOR_BG)
        self.geometry("520x760")
        self.resizable(False, False)
        
        # State
        self._is_processing = False
        self._rvc_engine = None
        self._model_path = None
        self._model_name = None
        self._output_dir = AUDIO_DIR
        self._input_audio_path = None
        
        # Audio Variables
        self.f0_key_var = tk.IntVar(value=0)
        self.method_var = tk.StringVar(value="rmvpe")
        self.index_rate_var = tk.DoubleVar(value=0.7)
        self.lang_var = tk.StringVar(value="Thai (th)")

        self._load_cached_settings()
        self._build_ui()
        self._auto_load_last_model()

    # --- Persistence Logic ---

    def _load_cached_settings(self):
        """Loads previously saved settings from local cache files."""
        out_file = os.path.join(MODEL_DIR, ".last_output")
        if os.path.isfile(out_file):
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    d = f.read().strip()
                if os.path.isdir(d):
                    self._output_dir = d
            except:
                pass

    def _save_setting(self, key, value):
        """Saves a setting to a hidden file in MODEL_DIR."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f".last_{key}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(value)

    # --- RVC Engine Logic ---

    def _load_rvc_model(self, pth_path):
        """Initializes RVC engine and loads the .pth model."""
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._log(f"Loading engine ({device})...", "dim")
            
            engine = RVCInference(device=device)
            engine.load_model(pth_path)
            
            self._rvc_engine = engine
            self._model_path = pth_path
            self._model_name = os.path.splitext(os.path.basename(pth_path))[0]
            
            self._save_setting("model", pth_path)
            self._update_status_indicator()
            self._log(f"Model loaded: {self._model_name}", "ok")
            return True
        except Exception as e:
            self._log(f"Error loading model: {e}", "err")
            return False

    def _auto_load_last_model(self):
        """Attempts to load the last used model on startup."""
        path_file = os.path.join(MODEL_DIR, ".last_model")
        if os.path.isfile(path_file):
            try:
                with open(path_file, "r", encoding="utf-8") as f:
                    path = f.read().strip()
                if os.path.isfile(path):
                    threading.Thread(target=self._load_rvc_model, args=(path,), daemon=True).start()
                    return
            except:
                pass
        self._update_status_indicator()

    # --- UI Components ---

    def _build_ui(self):
        """Constructs the high-fidelity dark-mode interface."""
        # Header
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=24, pady=(28, 0))
        
        tk.Label(header, text=APP_NAME, bg=COLOR_BG, fg=COLOR_ACCENT,
                 font=("Courier New", 28, "bold")).pack(side=tk.LEFT)
        
        self.status_dot = tk.Label(header, text="●", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 11))
        self.status_dot.pack(side=tk.RIGHT, pady=10)
        
        self.status_text = tk.Label(header, text="Checking...", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 9))
        self.status_text.pack(side=tk.RIGHT, padx=6, pady=10)
        
        tk.Label(self, text="Text to RVC Voice Interface", bg=COLOR_BG, fg=COLOR_SUBTEXT,
                 font=("Courier New", 9)).pack(anchor="w", padx=24)
        
        tk.Frame(self, bg=COLOR_DIM, height=1).pack(fill=tk.X, padx=24, pady=16)

        # 1. Model Selection
        row_model = tk.Frame(self, bg=COLOR_BG)
        row_model.pack(fill=tk.X, padx=24, pady=(0, 10))
        tk.Label(row_model, text="MODEL", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        
        self.btn_load_model = tk.Button(row_model, text="📂 Load .pth", command=self._browse_model,
                                        bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9),
                                        relief=tk.FLAT, borderwidth=0, padx=12, pady=4, cursor="hand2",
                                        activebackground=COLOR_DIM, activeforeground=COLOR_ACCENT)
        self.btn_load_model.pack(side=tk.RIGHT)
        
        self.lbl_model_name = tk.Label(row_model, text="No model selected", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 9))
        self.lbl_model_name.pack(side=tk.RIGHT, padx=8)

        # 2. Output Folder
        row_out = tk.Frame(self, bg=COLOR_BG)
        row_out.pack(fill=tk.X, padx=24, pady=(0, 10))
        tk.Label(row_out, text="OUTPUT", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        
        self.btn_change_out = tk.Button(row_out, text="📁 Change", command=self._browse_output_dir,
                                        bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9),
                                        relief=tk.FLAT, borderwidth=0, padx=12, pady=4, cursor="hand2",
                                        activebackground=COLOR_DIM, activeforeground=COLOR_ACCENT)
        self.btn_change_out.pack(side=tk.RIGHT)
        
        self.lbl_out_path = tk.Label(row_out, text=self._shorten_path(self._output_dir), bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 9))
        self.lbl_out_path.pack(side=tk.RIGHT, padx=8)

        # 3. Input Audio (Optional)
        row_in_audio = tk.Frame(self, bg=COLOR_BG)
        row_in_audio.pack(fill=tk.X, padx=24, pady=(8, 0))
        tk.Label(row_in_audio, text="SOURCE", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        
        self.lbl_source_audio = tk.Label(row_in_audio, text="-- TTS Mode --", bg=COLOR_BG, fg=COLOR_DIM, font=("Tahoma", 9))
        self.lbl_source_audio.pack(side=tk.LEFT, padx=(12, 0))

        self.btn_clear_audio = tk.Button(row_in_audio, text="✖ Clear", command=self._clear_input_audio,
                                         bg=COLOR_BG, fg=COLOR_RED, font=("Courier New", 8),
                                         relief=tk.FLAT, borderwidth=0, cursor="hand2", activebackground=COLOR_DIM)
        
        self.btn_browse_audio = tk.Button(row_in_audio, text="🎵 Browse Audio", command=self._browse_input_audio,
                                          bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9),
                                          relief=tk.FLAT, borderwidth=0, padx=12, pady=4, cursor="hand2",
                                          activebackground=COLOR_DIM, activeforeground=COLOR_ACCENT)
        self.btn_browse_audio.pack(side=tk.RIGHT)

        # 4. Text Input
        tk.Label(self, text="TEXT TO SPEAK", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(anchor="w", padx=24, pady=(12, 4))
        self.input_text = tk.Text(self, bg=COLOR_BG_SEC, fg=COLOR_TEXT, height=4,
                               font=("Tahoma", 10), insertbackground=COLOR_TEXT,
                               relief=tk.FLAT, padx=12, pady=12)
        self.input_text.pack(fill=tk.X, padx=24)
        self.input_text.bind("<Control-Return>", lambda e: self._on_speak_clicked())

        # 5. Advanced Settings
        row_settings = tk.Frame(self, bg=COLOR_BG)
        row_settings.pack(fill=tk.X, padx=24, pady=(14, 0))
        
        # Pitch
        tk.Label(row_settings, text="PITCH", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        tk.Spinbox(row_settings, from_=-24, to=24, textvariable=self.f0_key_var, width=3, bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=(6, 16))
        
        # Method
        tk.Label(row_settings, text="METHOD", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        method_cb = ttk.Combobox(row_settings, textvariable=self.method_var, values=["rmvpe", "harvest", "pm"], width=8, state="readonly")
        method_cb.pack(side=tk.LEFT, padx=(6, 16))
        
        # Index Rate
        tk.Label(row_settings, text="INDEX", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        tk.Scale(row_settings, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.index_rate_var, bg=COLOR_BG, fg=COLOR_TEXT, length=100, sliderrelief=tk.FLAT, highlightthickness=0).pack(side=tk.LEFT, padx=(6, 0))

        # 6. Language & Speak
        row_lang = tk.Frame(self, bg=COLOR_BG)
        row_lang.pack(fill=tk.X, padx=24, pady=(14, 0))
        tk.Label(row_lang, text="LANGUAGE (TTS)", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
        
        lang_menu = tk.OptionMenu(row_lang, self.lang_var, *LANGUAGES.keys())
        lang_menu.config(bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9),
                        activebackground=COLOR_DIM, activeforeground=COLOR_ACCENT,
                        relief=tk.FLAT, borderwidth=0, highlightthickness=0, indicatoron=0, padx=12)
        lang_menu["menu"].config(bg=COLOR_BG_TER, fg=COLOR_TEXT, font=("Courier New", 9), activebackground=COLOR_DIM, activeforeground=COLOR_ACCENT)
        lang_menu.pack(side=tk.RIGHT)

        tk.Frame(self, bg=COLOR_BG, height=16).pack()
        self.btn_speak = tk.Button(self, text="▶  SPEAK / CONVERT", command=self._on_speak_clicked,
                                   bg=COLOR_ACCENT, fg=COLOR_BG, font=("Courier New", 12, "bold"),
                                   relief=tk.FLAT, borderwidth=0, padx=0, pady=16, cursor="hand2",
                                   activebackground="#c8e832", activeforeground=COLOR_BG)
        self.btn_speak.pack(fill=tk.X, padx=24)

        # 7. Progress & Logs
        self.lbl_progress = tk.Label(self, text="", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 9))
        self.lbl_progress.pack(pady=(12, 0))
        
        self.progress_bg = tk.Frame(self, bg=COLOR_DIM, height=2)
        self.progress_bg.pack(fill=tk.X, padx=24, pady=(6, 0))
        self.progress_bar = tk.Frame(self.progress_bg, bg=COLOR_ACCENT, height=2, width=0)
        self.progress_bar.place(x=0, y=0, relheight=1)

        tk.Frame(self, bg=COLOR_DIM, height=1).pack(fill=tk.X, padx=24, pady=16)
        tk.Label(self, text="CONSOLE LOG", bg=COLOR_BG, fg=COLOR_SUBTEXT, font=("Courier New", 8)).pack(anchor="w", padx=24)
        
        log_frame = tk.Frame(self, bg=COLOR_BG_SEC, highlightbackground=COLOR_DIM, highlightthickness=1)
        log_frame.pack(fill=tk.X, padx=24, pady=(6, 0))
        
        self.text_log = tk.Text(log_frame, bg=COLOR_BG_SEC, fg=COLOR_SUBTEXT, height=6,
                                font=("Courier New", 9), relief=tk.FLAT,
                                state=tk.DISABLED, padx=12, pady=10, borderwidth=0)
        self.text_log.pack(fill=tk.X)
        self.text_log.tag_config("ok",  foreground=COLOR_GREEN)
        self.text_log.tag_config("err", foreground=COLOR_RED)
        self.text_log.tag_config("dim", foreground=COLOR_SUBTEXT)
        self.text_log.tag_config("hi",  foreground=COLOR_ACCENT)

        # Footer
        tk.Label(self, text="Ctrl+Enter to process | RVC is enabled when a model is loaded",
                 bg=COLOR_BG, fg="#333", font=("Courier New", 8)).pack(pady=(12, 8))

    # --- UI Actions ---

    def _update_status_indicator(self):
        """Updates the top-right status dot and label."""
        if self._rvc_engine and self._model_name:
            self.status_dot.config(fg=COLOR_GREEN)
            self.status_text.config(text=f"Engine: {self._model_name}", fg=COLOR_GREEN)
        else:
            self.status_dot.config(fg=COLOR_ORANGE)
            self.status_text.config(text="Model Ready", fg=COLOR_ORANGE)

    def _on_speak_clicked(self, event=None):
        """Triggers the TTS and RVC conversion process."""
        if self._is_processing:
            return
        
        text = self.input_text.get("1.0", tk.END).strip()
        if not self._input_audio_path and not text:
            self._log("Please provide text or an input audio file.", "err")
            return
        
        lang_code = LANGUAGES.get(self.lang_var.get(), "th")
        self._is_processing = True
        self.btn_speak.config(state=tk.DISABLED, text="⏳ Processing...", bg=COLOR_DIM)
        
        threading.Thread(target=self._process_pipeline, args=(text, lang_code), daemon=True).start()

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Open RVC Model (.pth)",
            filetypes=[("RVC Model", "*.pth"), ("All files", "*.*")],
            initialdir=MODEL_DIR
        )
        if not path:
            return
            
        self.lbl_model_name.config(text=os.path.basename(path), fg=COLOR_ORANGE)
        self.btn_load_model.config(state=tk.DISABLED, text="Loading...")
        
        threading.Thread(target=self._exec_load_model, args=(path,), daemon=True).start()

    def _exec_load_model(self, path):
        success = self._load_rvc_model(path)
        def finalize():
            self.btn_load_model.config(state=tk.NORMAL, text="📂 Load .pth")
            if success:
                self.lbl_model_name.config(text=self._model_name, fg=COLOR_GREEN)
            else:
                self.lbl_model_name.config(text="Failed to load", fg=COLOR_RED)
        self.after(0, finalize)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select Output Directory", initialdir=self._output_dir)
        if path:
            self._save_setting("output", path)
            self._output_dir = path
            self.lbl_out_path.config(text=self._shorten_path(path), fg=COLOR_GREEN)
            self._log(f"Output changed: {path}", "ok")

    def _browse_input_audio(self):
        path = filedialog.askopenfilename(
            title="Select Source Audio",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac *.m4a"), ("All files", "*.*")],
        )
        if path:
            self._input_audio_path = path
            self.lbl_source_audio.config(text=self._shorten_path(path), fg=COLOR_ACCENT)
            self.input_text.config(state=tk.DISABLED, bg=COLOR_BG)
            self.btn_clear_audio.pack(side=tk.LEFT, padx=8)

    def _clear_input_audio(self):
        self._input_audio_path = None
        self.lbl_source_audio.config(text="-- TTS Mode --", fg=COLOR_DIM)
        self.input_text.config(state=tk.NORMAL, bg=COLOR_BG_SEC)
        self.btn_clear_audio.pack_forget()

    # --- Processing Pipeline ---

    def _process_pipeline(self, text, lang):
        """Handles the multi-step audio generation pipeline."""
        try:
            timestamp = int(time.time() * 1000)
            raw_path = os.path.join(self._output_dir, f"tmp_{timestamp}.wav")
            final_path = os.path.join(self._output_dir, f"output_{timestamp}.wav")

            # Step 1: Pre-process
            if self._input_audio_path:
                self.after(0, self._log, f"Pre-processing source audio...", "hi")
                self.after(0, self._set_progress, "Normalizing audio...", 20)
                subprocess.run([
                    FFMPEG_PATH, "-y", "-i", self._input_audio_path,
                    "-ar", "32000", "-ac", "1", "-acodec", "pcm_s16le",
                    raw_path
                ], check=True, capture_output=True, startupinfo=SUBPROCESS_STARTUPINFO)
            else:
                self.after(0, self._log, f"Synthesizing [{lang}]: {text[:40]}...", "hi")
                self.after(0, self._set_progress, "Generating TTS...", 20)
                self._generate_tts(text, raw_path, lang)

            # Step 2: RVC Inference
            self.after(0, self._set_progress, "RVC Inference...", 55)
            if self._rvc_engine and self._exec_rvc_inference(raw_path, final_path):
                try: os.remove(raw_path)
                except: pass
                self.after(0, self._log, f"RVC Success ({self._model_name})", "ok")
            else:
                # Fallback to source if RVC is not available
                os.rename(raw_path, final_path)
                reason = "No model loaded" if not self._rvc_engine else "Inference failed"
                self.after(0, self._log, f"Warning: {reason}. Using raw output.", "err")

            # Step 3: Play & Clean
            self.after(0, self._set_progress, "Playing...", 85)
            play_audio(final_path)
            cleanup_old_audio()
            
            self.after(0, self._set_progress, "Done ✨", 100)
            self.after(0, self._log, f"Output saved: {os.path.basename(final_path)}", "ok")
            self.after(2000, lambda: self._set_progress("", 0))

        except Exception as e:
            self.after(0, self._log, f"Pipeline Error: {e}", "err")
            self.after(0, self._set_progress, "Failed", 0)
        finally:
            self.after(0, self._reset_ui)

    def _generate_tts(self, text, out_wav, lang):
        """Uses Edge-TTS for high-quality source speech."""
        mp3_tmp = out_wav.replace(".wav", ".mp3")
        voice_map = {
            "th": "th-TH-PremwadeeNeural",
            "en": "en-US-AriaNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "ko": "ko-KR-SunHiNeural",
            "fr": "fr-FR-DeniseNeural",
        }
        voice = voice_map.get(lang, "th-TH-PremwadeeNeural")
        
        # Edge-TTS call
        subprocess.run([
            sys.executable, "-m", "edge_tts", 
            "--text", text, 
            "--voice", voice, 
            "--write-media", mp3_tmp
        ], check=True, startupinfo=SUBPROCESS_STARTUPINFO, capture_output=True)

        # Convert for RVC (32kHz)
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", mp3_tmp,
            "-ar", "32000", "-ac", "1", "-acodec", "pcm_s16le",
            out_wav
        ], check=True, capture_output=True, startupinfo=SUBPROCESS_STARTUPINFO)
        
        try: os.remove(mp3_tmp)
        except: pass

    def _exec_rvc_inference(self, in_wav, out_wav):
        """Performs RVC voice conversion on the generated wav."""
        try:
            self._rvc_engine.set_params(
                f0up_key=self.f0_key_var.get(),
                f0method=self.method_var.get(), 
                index_rate=self.index_rate_var.get()
            )
            self._rvc_engine.infer_file(in_wav, out_wav)
            return os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1000
        except Exception as e:
            print(f"Inference error: {e}")
            return False

    # --- UI Helpers ---

    def _reset_ui(self):
        self._is_processing = False
        self.btn_speak.config(state=tk.NORMAL, text="▶  SPEAK / CONVERT", bg=COLOR_ACCENT)

    def _log(self, msg, tag="dim"):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n", tag)
        self.text_log.see(tk.END)
        self.text_log.config(state=tk.DISABLED)

    def _set_progress(self, msg, pct):
        self.lbl_progress.config(text=msg)
        width = self.progress_bg.winfo_width() or 472
        self.progress_bar.place(width=int(width * (pct / 100)))

    @staticmethod
    def _shorten_path(p, maxlen=40):
        if len(p) <= maxlen: return p
        return "..." + p[-(maxlen - 3):]

if __name__ == "__main__":
    app = RVCsayApp()
    app.mainloop()
