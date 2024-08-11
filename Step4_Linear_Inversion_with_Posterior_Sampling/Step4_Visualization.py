import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import numpy as np
from Forward_Tools.Gaussian_Pressure_Source import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, multivariate_normal


def plot_2prior_models(R, num_subs, prior_model1, prior_model2, title1="Uniform Prior Model:",
                       title2="Informative Prior Model:", file_name="2_prior_models.png"):
    """
    Plot the reservoir layer's horizontal profile for two prior models side by side.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param prior_model1: The 1D vector of the first prior model
    :param prior_model2: The 1D vector of the second prior model
    :param title1: The title of the first plot
    :param title2: The title of the second plot
    :param file_name: The file name to save the plot
    """
    # 1) Initialize the plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    def plot_model(ax, prior_model, title):
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        cmap = plt.get_cmap('jet')
        sub_pressures = [p / 1e6 for p in prior_model]  # Convert to MPa
        sub_radius = R / num_subs
        unique_pressures = np.unique(sub_pressures)

        if len(unique_pressures) > 10:
            norm = Normalize(vmin=min(sub_pressures), vmax=max(sub_pressures))
        else:
            if len(unique_pressures) > 1:
                bounds = np.concatenate((
                    [unique_pressures[0] - (unique_pressures[1] - unique_pressures[0]) / 2],
                    (unique_pressures[:-1] + unique_pressures[1:]) / 2,
                    [unique_pressures[-1] + (unique_pressures[-1] - unique_pressures[-2]) / 2]
                ))
            else:
                bounds = [unique_pressures[0] - 0.5, unique_pressures[0] + 0.5]

            cmap = ListedColormap(plt.cm.jet(np.linspace(0, 1, len(unique_pressures))))
            norm = BoundaryNorm(bounds, len(unique_pressures))

        for i in range(num_subs):
            for j in range(num_subs):
                x = -R + (2 * j + 1) * sub_radius
                y = -R + (2 * i + 1) * sub_radius
                index = i * num_subs + j
                circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
                ax.add_patch(circle)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(sub_pressures)

        if len(unique_pressures) > 10:
            cbar = plt.colorbar(sm, ax=ax)
        else:
            cbar = plt.colorbar(sm, ax=ax, ticks=unique_pressures, spacing='proportional')
            cbar.ax.set_yticklabels([f"{val:.1f}" for val in unique_pressures])

        cbar.set_label('Pressure [MPa]', fontsize=12)
        title = title + f"Standard Deviation is 1.2 MPa"
        ax.set_title(title, fontsize=14)

    # Plot the two models
    plot_model(axes[0], prior_model1, title1)
    plot_model(axes[1], prior_model2, title2)

    # 3) Save and show
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_true_model(R, num_subs, true_model, title="True Model 3: Pressure Plume with Impermeable Fault", file_name="true_model1.png"):
    """
    Plot the reservoir layer's horizontal profile for the true model.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param true_model: The 1D vector of the true model
    :param title: The title of the plot
    :param file_name: The file name to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    cmap = plt.get_cmap('jet')
    sub_pressures = [p / 1e6 for p in true_model]  # Convert to MPa
    sub_radius = R / num_subs
    unique_pressures = np.unique(sub_pressures)

    if len(unique_pressures) > 10:
        norm = Normalize(vmin=min(sub_pressures), vmax=max(sub_pressures))
    else:
        if len(unique_pressures) > 1:
            bounds = np.concatenate((
                [unique_pressures[0] - (unique_pressures[1] - unique_pressures[0]) / 2],
                (unique_pressures[:-1] + unique_pressures[1:]) / 2,
                [unique_pressures[-1] + (unique_pressures[-1] - unique_pressures[-2]) / 2]
            ))
        else:
            bounds = [unique_pressures[0] - 0.5, unique_pressures[0] + 0.5]

        cmap = ListedColormap(plt.cm.jet(np.linspace(0, 1, len(unique_pressures))))
        norm = BoundaryNorm(bounds, len(unique_pressures))

    for i in range(num_subs):
        for j in range(num_subs):
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
            ax.add_patch(circle)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(sub_pressures)

    if len(unique_pressures) > 10:
        cbar = plt.colorbar(sm, ax=ax)
    else:
        cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, spacing='proportional')
        tick_locs = (bounds[:-1] + bounds[1:]) / 2
        cbar.set_ticks(tick_locs)
        cbar.ax.set_yticklabels([f"{val:.1f}" for val in unique_pressures])

    cbar.set_label('Pressure [MPa]', fontsize=12)
    title = title + f"\n{num_subs ** 2} sub-reservoirs, the radius of each sub-reservoir is {R / num_subs:.1f}m"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined(X1, Y1, X2, Y2, uz_total, ur_total, tilt_x_total, tilt_y_total,
                  R, num_subs, p=None, sub_pressures=None,
                  num_levels=100, file_name='combined_plot.png'):
    """
    Plot observations and model in a combined figure with two subplots.
    :param X1: The meshgrid in x direction for tilt data
    :param Y1: The meshgrid in y direction for tilt data
    :param X2: The meshgrid in x direction for displacement data
    :param Y2: The meshgrid in y direction for displacement data
    :param uz_total: The vertical displacement at all observation points
    :param ur_total: The horizontal displacement at all observation points
    :param tilt_x_total: The horizontal tilt at all observation points
    :param tilt_y_total: The vertical tilt at all observation points
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param p: The pressure of the reservoir when num_subs == 0
    :param sub_pressures: The list of the sub-reservoirs pressures when num_subs != 0
    :param num_levels: The number of levels in the contour plot
    :param file_name: The file name to save the plot
    """
    fig = plt.figure(figsize=(35, 15))
    gs = gridspec.GridSpec(1, 2)

    # Create a 2x2 grid for observations on the right subplot
    gs_obs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.15, wspace=0.1)
    ax_obs_uz = fig.add_subplot(gs_obs[0, 0])
    ax_obs_ur = fig.add_subplot(gs_obs[0, 1])
    ax_obs_tiltx = fig.add_subplot(gs_obs[1, 0])
    ax_obs_tilty = fig.add_subplot(gs_obs[1, 1])

    # Plot Observations
    im1 = ax_obs_uz.contourf(X2, Y2, uz_total, levels=num_levels, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=ax_obs_uz)
    cbar1.set_label('Displacement [mm]', fontsize=14)
    ax_obs_uz.set_title('True uz', fontsize=18)
    ax_obs_uz.set_xlabel('X [m]', fontsize=14)
    ax_obs_uz.set_ylabel('Y [m]', fontsize=14)
    ax_obs_uz.tick_params(labelsize=12)

    im2 = ax_obs_ur.contourf(X2, Y2, ur_total, levels=num_levels, cmap='viridis')
    cbar2 = fig.colorbar(im2, ax=ax_obs_ur)
    cbar2.set_label('Displacement [mm]', fontsize=14)
    ax_obs_ur.set_title('True ur', fontsize=18)
    ax_obs_ur.set_xlabel('X [m]', fontsize=14)
    ax_obs_ur.set_ylabel('Y [m]', fontsize=14)
    ax_obs_ur.tick_params(labelsize=12)

    im3 = ax_obs_tiltx.contourf(X1, Y1, tilt_x_total, levels=num_levels, cmap='viridis')
    cbar3 = fig.colorbar(im3, ax=ax_obs_tiltx)
    cbar3.set_label('Tilt [microradian]', fontsize=14)
    ax_obs_tiltx.set_title('True tiltx', fontsize=18)
    ax_obs_tiltx.set_xlabel('X [m]', fontsize=14)
    ax_obs_tiltx.set_ylabel('Y [m]', fontsize=14)
    ax_obs_tiltx.tick_params(labelsize=12)

    im4 = ax_obs_tilty.contourf(X1, Y1, tilt_y_total, levels=num_levels, cmap='viridis')
    cbar4 = fig.colorbar(im4, ax=ax_obs_tilty)
    cbar4.set_label('Tilt [microradian]', fontsize=14)
    ax_obs_tilty.set_title('True tilty', fontsize=18)
    ax_obs_tilty.set_xlabel('X [m]', fontsize=14)
    ax_obs_tilty.set_ylabel('Y [m]', fontsize=14)
    ax_obs_tilty.tick_params(labelsize=12)

    # Plot Model on the left subplot
    ax_model = fig.add_subplot(gs[0])
    ax_model.set_xlim(-R, R)
    ax_model.set_ylim(-R, R)
    ax_model.set_xlabel('X [m]', fontsize=14)
    ax_model.set_ylabel('Y [m]', fontsize=14)
    ax_model.tick_params(labelsize=14)
    cmap = plt.get_cmap('jet')
    norm = Normalize()

    # Cond 1: No sub-reservoirs, only one reservoir (circle shape)
    if num_subs == 0:
        p = p / 1e6  # Convert to MPa
        norm.autoscale([p])
        circle = patches.Circle((0, 0), R, color=cmap(norm(p)), ec='black')
        ax_model.add_patch(circle)
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([p])
        cbar = plt.colorbar(sm, ax=ax_model)
        cbar.set_label('Pressure [MPa]', fontsize=14)
        ax_model.set_title(f'R={R}m Reservoir Model with no sub-reservoirs.\nThe pressure change is {p}Mpa',
                           fontsize=16)
    # Cond 2: With multiple sub-reservoirs
    else:
        sub_pressures = [p / 1e6 for p in sub_pressures]  # Convert to MPa
        sub_radius = R / num_subs
        unique_pressures = np.unique(sub_pressures)

        if len(unique_pressures) > 10:
            norm = Normalize(vmin=min(sub_pressures), vmax=max(sub_pressures))
        else:
            if len(unique_pressures) > 1:
                bounds = np.concatenate((
                    [unique_pressures[0] - (unique_pressures[1] - unique_pressures[0]) / 2],
                    (unique_pressures[:-1] + unique_pressures[1:]) / 2,
                    [unique_pressures[-1] + (unique_pressures[-1] - unique_pressures[-2]) / 2]
                ))
            else:
                bounds = [unique_pressures[0] - 0.5, unique_pressures[0] + 0.5]

            cmap = ListedColormap(plt.cm.jet(np.linspace(0, 1, len(unique_pressures))))
            norm = BoundaryNorm(bounds, len(unique_pressures))

        for i in range(num_subs):
            for j in range(num_subs):
                x = -R + (2 * j + 1) * sub_radius
                y = -R + (2 * i + 1) * sub_radius
                index = i * num_subs + j
                circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
                ax_model.add_patch(circle)

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(sub_pressures)
        if len(unique_pressures) > 10:
            cbar = plt.colorbar(sm, ax=ax_model)
        else:
            cbar = plt.colorbar(sm, ax=ax_model, boundaries=bounds, spacing='uniform')
            tick_locs = (bounds[:-1] + bounds[1:]) / 2
            cbar.set_ticks(tick_locs)
            cbar.ax.set_yticklabels([f"{val:.1f}" for val in unique_pressures])
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Pressure [MPa]', fontsize=16)
        ax_model.tick_params(labelsize=14)
        ax_model.set_title("True Model", fontsize=18)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    ###### Model Preparation ######
    # 1) Define the general parameters for the model
    model_general = {
        # Layer related parameters
        'model_flag': 1,
        'NL': 8,
        'NL_real': 8,
        'iL_pp': 6,  # The index of the reservoir layer
        'H': np.array([50, 422, 251.03, 246.52, 112.67, 70.21, 105.27, 50.3]),
        # Reservoir related parameters
        'R': 900,  # Reservoir's radius, unit [m]
        'P_reservoir': 2.4e6,  # Only for the G computation, not the real model.
        # Geo-mechanical parameters
        'Alpha': np.array([0.95, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]),
        'Nu': np.array([0.35, 0.33, 0.18, 0.25, 0.25, 0.2, 0.22, 0.25]),  # Poisson's ratio
        'G': np.array([0.15, 1.2, 10.48, 5.37, 10.23, 11.16, 10.23, 18]) * 1e9,  # Uint: Pa
        'RHO': np.array([]),
        'VS': np.array([[], []]),
        'VP': np.array([[], []])
    }
    # Further define the parameters: Self-adjusted, don't need to change there.
    # 1.1) Pressure in each layer
    model_general['P'] = np.zeros(model_general['NL'])
    model_general['P'][model_general['iL_pp'] - 1] = model_general['P_reservoir']
    # 1.2) The top and bottom depth of each layer, don't need to change there.
    model_general['Z_top'] = np.zeros(model_general['NL'])
    model_general['Z_bot'] = model_general['H']

    # 2) For tilt: larger spacing
    auxiliary_parameters_tilt = {
        'num_subs': 25,
        'r_max': 2 * model_general['R'],
        'dx': 400,
        'dy': 400,
        'process_num': 14,    # Change it according to the computer's configuration
        'print_info': False
    }
    auxiliary_parameters_disp = {
        'num_subs': 25,
        'r_max': 2 * model_general['R'],
        'dx': 50,
        'dy': 50,
        'process_num': 14,  # Change it according to the computer's configuration
        'print_info': False
    }

    model_prior1 = np.full(625, 2.4e6)
    centers = [(12, 12)]
    sigmas = [(7, 7)]   # x/y radius
    amplitudes = [2.4e6]  # Different amplitudes for different regions: max = 8Mpa
    zero_zone_vertices = [(6, 0), (24, 20), (24, 0)]
    model_prior2 = np.array(assign_gaussian_pressure_with_zero_zone(num_subs=auxiliary_parameters_tilt['num_subs'],
                                                                    centers=centers, sigmas=sigmas, amplitudes=amplitudes, zero_zone_vertices=zero_zone_vertices))
    plot_2prior_models(R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], prior_model1=model_prior1, prior_model2=model_prior2)

    true_model = np.load("model_saved.npy").flatten()
    plot_true_model(R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], true_model=true_model)


    ## True model + True observation in one figure ###
    # tilt
    x1 = np.arange(-auxiliary_parameters_tilt['r_max'], auxiliary_parameters_tilt['r_max'] + auxiliary_parameters_tilt['dx'], auxiliary_parameters_tilt['dx'])
    y1 = np.arange(-auxiliary_parameters_tilt['r_max'], auxiliary_parameters_tilt['r_max'] + auxiliary_parameters_tilt['dy'], auxiliary_parameters_tilt['dy'])
    X1, Y1 = np.meshgrid(x1, y1)
    # disp
    x2 = np.arange(-auxiliary_parameters_disp['r_max'], auxiliary_parameters_disp['r_max'] + auxiliary_parameters_disp['dx'], auxiliary_parameters_disp['dx'])
    y2 = np.arange(-auxiliary_parameters_disp['r_max'], auxiliary_parameters_disp['r_max'] + auxiliary_parameters_disp['dy'], auxiliary_parameters_disp['dy'])
    X2, Y2 = np.meshgrid(x2, y2)
    # G
    G_tiltx = np.load("G_tiltx_100x625.npy")
    G_tilty = np.load("G_tilty_100x625.npy")
    G_uz = np.load("G_uz_5329x625.npy")
    G_ur = np.load("G_ur_5329x625.npy")
    true_model = np.load("model_saved.npy").flatten()
    # data in 1D vector
    true_tiltx = G_tiltx @ true_model
    true_tilty = G_tilty @ true_model
    true_uz = G_uz @ true_model
    true_ur = G_ur @ true_model
    # reshape data in 2D matrix
    uz_total = true_uz.reshape(X2.shape)
    ur_total = true_ur.reshape(X2.shape)
    tilt_x_total = true_tiltx.reshape(X1.shape)
    tilt_y_total = true_tilty.reshape(X1.shape)
    # Plot
    plot_combined(X1=X1, Y1=Y1, X2=X2, Y2=Y2,
                  uz_total=uz_total, ur_total=ur_total, tilt_x_total=tilt_x_total, tilt_y_total=tilt_y_total,
                  R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], sub_pressures=true_model, file_name='true_model_obs_in1fig.png')
