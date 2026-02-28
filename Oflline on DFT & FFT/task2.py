import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, FastFourierTransform, DiscreteSignal


class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")

        self.samplerate = 0
        self.original_audio = None
        self.processed_audio = None

        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        tk.Button(top_frame, text="Load WAV File",
                  command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play",
                  command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio",
                  command=sd.stop).pack(side=tk.LEFT, padx=10)

        # Toggle Switch
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text="Algorithm: ").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="DFT (Slow)",
                       variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT (Fast)",
                       variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # Equalizer Sliders
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=20, padx=20)

        self.sliders = []
        labels = ["Low", "Low-Mid", "Mid", "High-Mid", "High"]
        for i in range(5):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0,
                              resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                self.samplerate, data = wav.read(file_path)

                # Normalize to float [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0

                # If already float, just ensure float32
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                self.original_audio = data
                self.processed_audio = None
                duration = len(data) / self.samplerate
                print(
                    f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def getBandIndices(self, N, sampleRate):
        freqs = [0, 300, 1000, 4000, 8000, sampleRate//2]
        indices = []
        for i in range(len(freqs)-1):
            low = int(freqs[i]*N/sampleRate)
            high = int(freqs[i+1]*N/sampleRate)
            indices.append((low, high))
        return indices

    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return

        self.root.update()

        print("Starting processing...")
        # Get Slider Values
        gains = [s.get() for s in self.sliders]

        if self.use_fft.get():
            analyzer = FastFourierTransform()
            chunkSize = 2048
        else:
            analyzer = DFTAnalyzer()
            chunkSize = 1024

        audio = self.original_audio
        totalSamples = len(audio)
        output = np.zeros(totalSamples, dtype=np.float32)
        bandIndices = self.getBandIndices(chunkSize, self.samplerate)
        currPos = 0
        while currPos < totalSamples:
            end = min(chunkSize+currPos, totalSamples)
            chunk = audio[currPos:end].astype(np.complex128)

            actualLen = len(chunk)
            if actualLen < chunkSize:
                chunk = np.pad(chunk, (0, chunkSize-actualLen))
            signal = DiscreteSignal(chunk)
            spectrum = analyzer.compute_dft(signal)

            filtered = spectrum.copy()
            for bandIndex, (low, high) in enumerate(bandIndices):
                gain = gains[bandIndex]
                filtered[low:high] *= gain
                if low > 0:
                    filtered[chunkSize-high:chunkSize-low] *= gain

            reconstructed = analyzer.compute_idft(filtered)
            outputChunk = reconstructed.real[:actualLen].astype(np.float32)
            output[currPos:currPos+actualLen] = outputChunk

            currPos += chunkSize

        maxVal = np.max(np.abs(output))
        if maxVal > 1.0:
            output /= maxVal

        self.processed_audio = output

        sd.stop()
        default_output = sd.default.device[0]
        sd.play(self.processed_audio, self.samplerate, device=default_output)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()
