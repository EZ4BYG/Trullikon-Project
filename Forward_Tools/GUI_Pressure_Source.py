import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
# GUI input sub-reservoirs
import tkinter as tk
from tkinter import simpledialog


def set_sub_pressures(num_subs, max_p=20):
    root = tk.Tk()
    root.title("Set Sub-Reservoir Pressures")

    pressures = [0] * (num_subs ** 2)
    norm = Normalize(vmin=0, vmax=max_p)

    def get_color(pressure):
        # Color mapping
        cmap = plt.get_cmap('viridis')
        rgba_color = cmap(norm(pressure))
        return to_hex(rgba_color)

    def on_button_click(i):
        pressure = simpledialog.askfloat("Input", f"Set pressure for sub-reservoir {i + 1} (MPa):", parent=root,
                                         minvalue=0, maxvalue=max_p)
        if pressure is not None:
            pressures[i] = pressure * 1e6  # Convert to Pa
            buttons[i].config(text=f"{pressure} MPa", bg=get_color(pressure))

    # Create buttons
    buttons = []
    for i in range(num_subs ** 2):
        # Note there: row-major order!
        row = num_subs - 1 - (i // num_subs)
        column = i % num_subs
        button = tk.Button(root, text="0 MPa", command=lambda i=i: on_button_click(i), bg=get_color(0))
        button.grid(row=row, column=column, sticky='nsew', padx=5, pady=5)
        buttons.append(button)

    for i in range(num_subs):
        root.grid_rowconfigure(i, weight=1)
        root.grid_columnconfigure(i, weight=1)

    root.mainloop()
    print("Sub-reservoir pressures are set successfully!")
    return pressures