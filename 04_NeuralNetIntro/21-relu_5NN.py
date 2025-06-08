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
        self.w0_var = np.array([tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0)])
        self.b0_var = np.array([tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)])
        self.w1_var = np.array([tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0)])
        self.b1_var = np.array([tk.DoubleVar(value=0.0)])

        self.setup_controls()
        self.setup_plot()
        self.update_plot()

    def setup_controls(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, expand=True)

        self.w0_slider = []
        self.b0_slider = []
        self.w1_slider = []
        self.b1_slider = []
        for k in range(0,5):
            ttk.Label(frame, text=f"w0{k}:").grid(row=k, column=0, sticky='w')
            self.w0_slider.append(ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.w0_var[k], orient='horizontal', command=self.on_slider_move, length=100))
            self.w0_slider[-1].grid(row=k, column=1, sticky='ew', padx=1)
            ttk.Label(frame, text=f"b0{k}:").grid(row=k, column=2, sticky='w')
            self.b0_slider.append(ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b0_var[k], orient='horizontal', command=self.on_slider_move, length=100))
            self.b0_slider[-1].grid(row=k, column=3, sticky='ew', padx=1)
        for k in range(0,5):
            ttk.Label(frame, text=f"w1{k}:").grid(row=5+k, column=0, sticky='w')
            self.w1_slider.append(ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.w1_var[k], orient='horizontal', command=self.on_slider_move, length=100))
            self.w1_slider[-1].grid(row=5+k, column=1, sticky='ew', padx=1)
        ttk.Label(frame, text=f"b10:").grid(row=9, column=2, sticky='w')
        self.b1_slider.append(ttk.Scale(frame, from_=-5.0, to=5.0, variable=self.b1_var[0], orient='horizontal', command=self.on_slider_move, length=100))
        self.b1_slider[-1].grid(row=9, column=3, sticky='ew', padx=1)


        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        w0 = np.array([obj.get() for obj in self.w0_var])
        b0 = np.array([obj.get() for obj in self.b0_var])
        w1 = np.array([obj.get() for obj in self.w1_var])
        b1 = np.array([self.b1_var[0].get()])

        x = np.linspace(-10, 10, 400)
        y = np.array([w1[k]*relu(w0[k] * x + b0[k]) for k in range(0,5)])

        self.ax.clear()
        for k in range(0,5):
            self.ax.plot(x, y[k], '--', label=f"h{k}")
        self.ax.plot(x, sum(y)+b1, label=f"H")
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
