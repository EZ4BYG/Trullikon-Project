import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class InteractiveGrid:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Grid Selection")

        # Layout frames
        self.control_frame = tk.Frame(master, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create input for grid size and generator button
        ttk.Label(self.control_frame, text="X-axis points:").pack(padx=10, pady=5)
        self.x_entry = tk.Entry(self.control_frame, width=7)
        self.x_entry.pack(padx=10, pady=5)
        self.x_entry.insert(0, "25")  # Default value

        ttk.Label(self.control_frame, text="Y-axis points:").pack(padx=10, pady=5)
        self.y_entry = tk.Entry(self.control_frame, width=7)
        self.y_entry.pack(padx=10, pady=5)
        self.y_entry.insert(0, "25")  # Default value

        self.generate_button = tk.Button(self.control_frame, text="Generate", command=self.generate_grid)
        self.generate_button.pack(padx=10, pady=10)

        # Value entry and set button
        ttk.Label(self.control_frame, text="Set Value:").pack(padx=10, pady=5)
        self.value_entry = tk.Entry(self.control_frame, width=7)
        self.value_entry.pack(padx=10, pady=5)

        self.set_button = tk.Button(self.control_frame, text="Set Value", command=self.set_value)
        self.set_button.pack(padx=10, pady=10)

        # Fill unassigned entry and button
        ttk.Label(self.control_frame, text="Fill Unassigned:").pack(padx=10, pady=5)
        self.fill_value_entry = tk.Entry(self.control_frame, width=7)
        self.fill_value_entry.pack(padx=10, pady=5)

        self.fill_button = tk.Button(self.control_frame, text="Fill Unassigned", command=self.fill_unassigned)
        self.fill_button.pack(padx=10, pady=10)

        # Create a figure and a canvas, initially empty
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        # Initialize plot settings
        self.ax.set_title("Interactive Grid Model Generation")
        self.ax.set_xlabel("X-Axis Grid")
        self.ax.set_ylabel("Y-Axis Grid")

        self.data = None
        self.lasso = None

    def generate_grid(self):
        x_num = int(self.x_entry.get())
        y_num = int(self.y_entry.get())
        self.data = np.full((y_num, x_num), np.nan)
        self.cmap = plt.get_cmap('viridis')
        self.norm = plt.Normalize()
        self.ax.clear()
        self.im = self.ax.imshow(self.data, cmap=self.cmap, norm=self.norm, interpolation='nearest', origin='lower')
        self.ax.scatter(np.repeat(np.arange(x_num), y_num), np.tile(np.arange(y_num), x_num), color='red', s=10)  # Show points
        if self.lasso:
            self.lasso.disconnect_events()
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.canvas.draw_idle()

    def onselect(self, verts):
        path = Path(verts)
        x, y = np.meshgrid(np.arange(self.data.shape[1]), np.arange(self.data.shape[0]))
        points = np.hstack((x.flatten()[:, np.newaxis], y.flatten()[:, np.newaxis]))
        self.mask = path.contains_points(points).reshape(self.data.shape)
        self.canvas.draw_idle()

    def set_value(self):
        if self.mask is not None:
            try:
                value = float(self.value_entry.get())
                self.data[self.mask] = value
                self.norm.autoscale(np.nan_to_num(self.data))
                self.im.set_data(self.data)
                self.canvas.draw_idle()
            except ValueError:
                pass

    def fill_unassigned(self):
        try:
            fill_value = float(self.fill_value_entry.get())
            unassigned_mask = np.isnan(self.data)
            self.data[unassigned_mask] = fill_value
            self.im.set_data(self.data)
            self.canvas.draw_idle()
        except ValueError:
            pass

    def save_data(self, filename='model_saved.npy'):
        """ Save the data array to a file. """
        np.save(filename, self.data)
        print("True Model Saved Successfully!")


if __name__ == '__main__':
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()
    app.save_data()  # Save the data when the GUI is closed
