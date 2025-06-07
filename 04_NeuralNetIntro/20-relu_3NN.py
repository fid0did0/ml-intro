import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

def relu(x):
    rl=np.maximum(x,0)
    return rl

class LinearPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive y = bx + c Plot")

        # Set up sliders for b and c
        self.w0_var = np.array([tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0)])
        self.b0_var = np.array([tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)])

        self.setup_controls()
        self.setup_plot()
        self.update_plot()

    def setup_controls(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, expand=True)

        ttk.Label(frame, text="w00:").grid(row=0, column=0, sticky='w')
        self.w00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.w0_var[0], orient='horizontal', command=self.on_slider_move, length=100)
        self.w00_slider.grid(row=0, column=1, sticky='ew', padx=1)
        ttk.Label(frame, text="b00:").grid(row=0, column=2, sticky='w')
        self.b00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b0_var[0], orient='horizontal', command=self.on_slider_move, length=100)
        self.b00_slider.grid(row=0, column=3, sticky='ew', padx=1)

        ttk.Label(frame, text="w01:").grid(row=1, column=0, sticky='w')
        self.w00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.w0_var[1], orient='horizontal', command=self.on_slider_move, length=100)
        self.w00_slider.grid(row=1, column=1, sticky='ew', padx=1)
        ttk.Label(frame, text="b01:").grid(row=1, column=2, sticky='w')
        self.b00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b0_var[1], orient='horizontal', command=self.on_slider_move, length=100)
        self.b00_slider.grid(row=1, column=3, sticky='ew', padx=1)

        ttk.Label(frame, text="w02:").grid(row=2, column=0, sticky='w')
        self.w00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.w0_var[2], orient='horizontal', command=self.on_slider_move, length=100)
        self.w00_slider.grid(row=2, column=1, sticky='ew', padx=1)
        ttk.Label(frame, text="b02:").grid(row=2, column=2, sticky='w')
        self.b00_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b0_var[2], orient='horizontal', command=self.on_slider_move, length=100)
        self.b00_slider.grid(row=2, column=3, sticky='ew', padx=1)

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        w0 = np.array([self.w0_var[0].get(), self.w0_var[1].get(), self.w0_var[2].get()])
        b0 = np.array([self.b0_var[0].get(), self.b0_var[1].get(), self.b0_var[2].get()])

        x = np.linspace(-10, 10, 400)
        y = np.array([relu(w0[0] * x + b0[0]), relu(w0[1] * x + b0[1]), relu(w0[2] * x + b0[2])])

        self.ax.clear()
        self.ax.plot(x, y[0], label=f"h0")
        self.ax.plot(x, y[1], label=f"h1")
        self.ax.plot(x, y[2], label=f"h2")
        self.ax.plot(x, sum(y), label=f"H")
        self.ax.axhline(0, color='gray', lw=1)
        self.ax.axvline(0, color='gray', lw=1)
        self.ax.grid(True)
        self.ax.set_ylim(-20, 20)
        #self.ax.set_title("Plot of y = w00x + b00")
        #self.ax.legend()
        self.canvas.draw()

    def on_slider_move(self, event):
        self.update_plot()
    
    def on_closing(self):
        self.root.quit()   # Exit the main loop
        self.root.destroy()  # Destroy the window and clean up

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearPlotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Graceful exit
    root.mainloop()
