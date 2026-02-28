import tkinter as tk
import numpy as np
import math
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierTransform


class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")

        # --- UI Layout ---
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        # Buttons
        tk.Button(control_frame, text="Clear Canvas",
                  command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles",
                  command=self.run_transform).pack(side=tk.LEFT, padx=5)

        # Toggle Switch (Radio Buttons)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(
            side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT",
                       variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT",
                       variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points = []
        self.drawing = False
        self.fourier_coeffs = None
        self.is_animating = False
        self.after_id = None
        self.pathPoints = []

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.drawing = True
        self.pathPoints = []

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(
                x-r, y-r, x+r, y+r, fill="black", outline="black")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.pathPoints = []
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    def draw_epicycle(self, x, y, radius):
        """
        Helper method for students to draw a circle (epicycle).
        x, y: Center coordinates
        radius: Radius of the circle
        """
        self.canvas.create_oval(
            x-radius, y-radius, x+radius, y+radius, outline="blue", tags="epicycle")

    def run_transform(self):
        if len(self.points) < 2:
            return

        # TODO: Implementation
        # 1. Convert (x,y) points to Complex Signal
        complexData = np.array(
            [x+1j*y for x, y in self.points], dtype=np.complex128)
        complexData = complexData - np.mean(complexData)
        signal = DiscreteSignal(complexData)
        # 2. Select Algorithm
        if self.use_fft.get():
            analyzer = FastFourierTransform()
            N = len(signal)
            power = 1
            while power < N:
                power <<= 1
            if power != N:
                signal = signal.interpolate(power)
        else:
            analyzer = DFTAnalyzer()
        # 3. Compute Transform
        self.fourier_coeffs = analyzer.compute_dft(signal)
        self.num_frames = len(self.fourier_coeffs)

        order = np.argsort(-np.abs(self.fourier_coeffs))
        self.sortedCoeffs = self.fourier_coeffs[order]
        self.sortedFreqs = order

        # setting canvas center
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        centerX = (min(xs) + max(xs)) / 2
        centerY = (min(ys) + max(ys)) / 2
        self.center_offset = (centerX, centerY)

        # canvas clearing for animation
        self.canvas.delete("all")
        self.pathPoints = []
        self.animate_epicycles(self.center_offset)

    def animate_epicycles(self, center_offset):
        self.is_animating = True
        self.time_step = 0
        # self.num_frames = ...

        self.center_offset = center_offset
        self.update_frame()

    def update_frame(self):
        if not self.is_animating:
            return

        self.canvas.delete("epicycle")

        N = self.num_frames
        t = self.time_step

        cx, cy = self.center_offset

        for i, (coeff, freq) in enumerate(zip(self.sortedCoeffs, self.sortedFreqs)):
            radius = np.abs(coeff)/N
            if radius < .5:
                continue
            angle = (2*np.pi*freq*t/N) + np.angle(coeff)
            self.draw_epicycle(cx, cy, radius)
            nx = cx + radius*np.cos(angle)
            ny = cy + radius*np.sin(angle)
            self.canvas.create_line(
                cx, cy, nx, ny, fill="gray", tags="epicycle")
            cx, cy = nx, ny

        self.pathPoints.append((cx, cy))

        if len(self.pathPoints) > 1:
            x0, y0 = self.pathPoints[-2]
            x1, y1 = self.pathPoints[-1]
            self.canvas.create_line(
                x0, y0, x1, y1, fill="red", width=2, tags="path")

        self.time_step = (self.time_step + 1) % N
        if self.time_step == 0:
            # self.canvas.delete("epicycle")
            self.pathPoints = []

        self.after_id = self.root.after(50, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()
