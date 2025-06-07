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
        self.b_var = tk.DoubleVar(value=1.0)
        self.c_var = tk.DoubleVar(value=0.0)

        self.setup_controls()
        self.setup_plot()
        self.update_plot()

    def setup_controls(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, text="b:").grid(row=0, column=0, sticky='w')
        self.b_slider = ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b_var, orient='horizontal', command=self.on_slider_move)
        self.b_slider.grid(row=0, column=1, sticky='ew', padx=5)

        ttk.Label(frame, text="c:").grid(row=1, column=0, sticky='w')
        self.c_slider = ttk.Scale(frame, from_=-10.0, to=10.0, variable=self.c_var, orient='horizontal', command=self.on_slider_move)
        self.c_slider.grid(row=1, column=1, sticky='ew', padx=5)

        frame.columnconfigure(1, weight=1)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        b = self.b_var.get()
        c = self.c_var.get()

        x = np.linspace(-10, 10, 400)
        y = relu(b * x + c)

        self.ax.clear()
        self.ax.plot(x, y, label=f"y = {b:.2f}x + {c:.2f}")
        self.ax.axhline(0, color='gray', lw=1)
        self.ax.axvline(0, color='gray', lw=1)
        self.ax.grid(True)
        self.ax.set_ylim(-20, 20)
        self.ax.set_title("Plot of y = bx + c")
        self.ax.legend()
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
