# RVCsay 🎙️

**RVCsay** is a lightweight, high-performance GUI wrapper for **Edge-TTS** and **RVC (Retrieval-based Voice Conversion)**. It allows you to transform text or existing audio into any voice model seamlessly on Windows and Linux—**WITHOUT** needing a separate RVC server.

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue.svg)](https://github.com/YOUR_USERNAME/RVCsay)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- **High-Fidelity Audio**: Powered by `Edge-TTS` for natural-sounding source speech.
- **Local RVC Inference**: No cloud or private servers required; runs directly on your machine.
- **Dual Mode**: Support for **Text-to-Voice (TTS)** and **Audio-to-Audio (Voice Conversion)**.
- **Advanced Parameters**: Fine-tune Pitch, Inference Method (RMVPE, Harvest, PM), and Index Rate via a sleek dark-mode GUI.
- **GPU Acceleration**: NVIDIA GPU support (CUDA) for near-instant processing.

---

## 🚀 Installation

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg** (Must be in your system PATH)

### 2. Setup (Windows/Linux)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RVCsay.git
cd RVCsay

# Install dependencies
pip install -r requirements.txt

# (Optional) NVIDIA GPU Support (Recommended for speed)
# See: https://pytorch.org/get-started/locally/
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** If `ffmpeg` is not in your system path, the application will attempt to find it in common local directories automatically.

---

## 🎯 Usage

Launch the GUI:
```bash
python RVCsay_gui.py
```

1. **Load Model**: Click `📂 Load .pth` and select your RVC voice model.
2. **Select Mode**:
    - **TTS**: Type your text in the box and select a language.
    - **Voice Conversion**: Click `🎵 Browse Audio` to use an existing file.
3. **Configure**: Adjust Pitch (+12 for male-to-female, -12 for female-to-male) and Method.
4. **Speak**: Click `▶ SPEAK / CONVERT` and enjoy the result!

---

## 📁 Project Structure

```text
RVCsay/
├── RVCsay_gui.py   # Main Application (Tkinter + RVC Engine)
├── README.md       # Project Documentation
└── ~/rvc-voice/    # Local storage for audio and models (Auto-created)
    ├── models/     # Store your .pth models here
    └── audio/      # Generated audio outputs
```

---

## 🗣️ Supported Languages (TTS)

| Code | Language | Code | Language |
|------|----------|------|----------|
| `th` | Thai     | `en` | English  |
| `ja` | Japanese | `zh` | Chinese  |
| `ko` | Korean   | `fr` | French   |

---

## 🇹🇭 สำหรับผู้ใช้งานภาษาไทย

**RVCsay** คือเครื่องมือแปลงข้อความเป็นเสียงที่ใช้เทคโนโลยี RVC เพื่อให้ได้เสียงที่มีเอกลักษณ์และคุณภาพสูง โดยไม่ต้องรันเซิร์ฟเวอร์แยก!

- **จุดเด่น**: ใช้ Edge-TTS ทำให้เสียงต้นฉบับชัดเจนกว่า gTTS ทั่วไป
- **การติดตั้ง**: ทำตามขั้นตอนด้านบน (Requirements) และติดตั้ง `ffmpeg` ให้เรียบร้อย
- **การใช้งาน**: โหลดไฟล์โมเดล `.pth`, พิมพ์ข้อความภาษาไทย, แล้วกด Speak ได้เลย!

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use and contribute!

---

*Made with ❤️ for the AI Voice Community.*
