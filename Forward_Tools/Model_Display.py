import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_model(R, num_subs, p=None, sub_pressures=None, file_name='reservoir_model.png'):
    """
    PLot the reservoir layer's horizontal profile.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param p: The pressure of the reservoir when num_subs == 0
    :param sub_pressures: The list of the sub-reservoirs pressures when num_subs != 0
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    cmap = plt.get_cmap('jet')
    norm = Normalize()

    # Cond 1: No sub-reservoirs, only one reservoir (circle shape)
    if num_subs == 0:
        p = p / 1e6  # Convert to MPa
        norm.autoscale([p])
        circle = patches.Circle((0, 0), R, color=cmap(norm(p)), ec='black')
        ax.add_patch(circle)
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([p])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pressure [MPa]', fontsize=12)
        ax.set_title(f'R={R}m Reservoir Model with no sub-reservoirs.\nThe pressure change is {p}Mpa', fontsize=14)
    # Cond 2: With multiple sub-reservoirs
    else:
        sub_pressures = [p / 1e6 for p in sub_pressures]  # Convert to MPa
        sub_radius = R / num_subs
        norm.autoscale(sub_pressures)
        for i in range(num_subs):
            for j in range(num_subs):
                # Note there: row-major order! so x uses j, y uses i.
                x = -R + (2 * j + 1) * sub_radius
                y = -R + (2 * i + 1) * sub_radius
                index = i * num_subs + j
                circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
                ax.add_patch(circle)
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(sub_pressures)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pressure [MPa]', fontsize=12)
        ax.set_title(f'Horizontal profile: Reservoir Model with {num_subs ** 2} sub-reservoirs\nThe radius of each sub-reservoir is {R/num_subs:.1f}m', fontsize=14)

    # Save and show
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    # return the figure
    return ax


def plot_model_original(model_general, r_max, dr, file_name='reservoir_model_original.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    # 0) Extract related model parameters
    H = model_general['H']
    iL_pp = model_general['iL_pp']
    r = np.arange(0, r_max + dr, dr)
    R = model_general['R']
    NL_real = model_general['NL']
    model_flag = model_general['model_flag']
    if model_flag == 1:
        # Isotropic model
        parameter_to_visualize = model_general['G'] / 1e9  # Convert to GPa
    else:
        # Anisotropic model
        parameter_to_visualize = model_general['VS']

    # 1) Determine unique values and create a discrete colormap
    unique_parameters = np.unique(parameter_to_visualize)
    unique_values = len(unique_parameters)
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, unique_values)))
    # Calculate segment lengths proportional to the layer thicknesses
    segment_lengths = np.array([H[np.where(parameter_to_visualize == p)].sum() for p in unique_parameters])
    segment_lengths = segment_lengths / np.sum(segment_lengths)
    # Create boundaries for each unique parameter value according to the segment lengths
    bounds = np.zeros(unique_values + 1)
    bounds[1:] = np.cumsum(segment_lengths)
    norm = BoundaryNorm(bounds, cmap.N)
    # depths
    depth = 0
    depths = [0]  # depth of each layer's top

    # 2) Plot the model
    if H[-1] == np.inf:
        H[-1] = np.max(H[:-1]) * 0.5  # Specify a value for the last infinite layer, only for better visualization
        # 2.1) Plot each layer
        for i, height in enumerate(H):
            color = cmap(norm(bounds[np.where(parameter_to_visualize[i] == unique_parameters)[0][0]]))  # Determine the color
            ax.fill_betweenx([-depth - height, -depth], -r[-1], r[-1], color=color)  # Draw the layer
            if i == (iL_pp - 1):  # Add a slash shadow in the reservoir layer
                ax.fill_betweenx([-depth - height, -depth], -R, R,
                                 color=color, alpha=0.5, hatch='/', edgecolor='k', linewidth=2)
            depth += height
            depths.append(-depth)
        # Setup y-axis ticks and labels
        yticks = depths[:]
        ax.set_yticks(yticks)
        yticklabels = ax.set_yticklabels([f"{d:.2f}" if d != depths[-1] else '-∞' for d in depths])
        yticklabels[-1].set_fontsize(12)
    else:
        # 2.2) Plot each layer
        for i, height in enumerate(H):
            color = cmap(norm(bounds[np.where(parameter_to_visualize[i] == unique_parameters)[0][0]]))  # Determine the color
            ax.fill_betweenx([-depth - height, -depth], -r[-1], r[-1], color=color)  # Draw the layer
            if i == (iL_pp - 1):  # Add a slash shadow in the reservoir layer
                ax.fill_betweenx([-depth - height, -depth], -R, R,
                                 color=color, alpha=0.5, hatch='/', edgecolor='k', linewidth=2)
            depth += height
            depths.append(-depth)
        # Setup y-axis ticks and labels
        yticks = depths[:]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{d:.2f}" for d in depths])
    # 2.3) Setup image properties
    ax.set_xlim(-r[-1], r[-1])
    ax.set_ylim(-sum(H), 0)
    ax.set_xlabel('Observation Radial Distance [m]', fontsize=12)
    ax.set_ylabel('Depth [m]', fontsize=12)
    if model_flag == 1:
        # Isotropic model
        ax.set_title(
            f'Vertical profile: Isotropic {NL_real}-layer Geological Model for Trüllikon \nTarget Formation: Radius={R}m, Thickness={H[iL_pp-1]}m, Depth={np.sum(H[:iL_pp-1]):.1f}m',
            fontsize=14)
    else:
        # Anisotropic model
        ax.set_title(
            f'Vertical profile: Anisotropic (VTI) {NL_real}-layer Geological Model for Trüllikon \nTarget Formation: Radius={R}m, Thickness={H[iL_pp-1]}m, Depth={np.sum(H[:iL_pp-1]):.1f}m',
            fontsize=14)
    ax.set_aspect('auto')
    # 2.4) Setup x-axis ticks and labels
    xticks = np.linspace(-r[-1], r[-1], num=9)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}" for x in np.abs(xticks)])
    # 2.5) Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, boundaries=bounds, ticks=(bounds[:-1] + bounds[1:]) / 2, spacing='proportional')
    cbar.set_ticklabels([f"{param:.2f}" for param in unique_parameters])
    if model_flag == 1:
        # Isotropic model
        cbar.set_label('Shear Modulus [GPa]', fontsize=12)
    else:
        # Anisotropic model
        cbar.set_label('Horizontal S-wave Velocity [m/s]', fontsize=12)
    # cbar.ax.tick_params(labelsize=12)

    # 3) Save and show
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

    # return the figure
    return ax


def plot_model_merge(ax1, ax2, file_name='reservoir_model_merge.png'):
    pass

