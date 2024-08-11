import time
from math import ceil
import numpy as np
# 1) Functions to assign a Gaussian pressure source:
# Two options: 1) assign_generalized_gaussian_pressure; 2) assign_gaussian_pressure
from Forward_Tools.Generalized_Gaussian_Pressure_Source import assign_generalized_gaussian_pressure
from Forward_Tools.Gaussian_Pressure_Source import assign_gaussian_pressure
# 2) Functions for the forward computation:
from Forward_Tools.Observations_Computation import grid_1d_model, compute_1st_gradients, compute_2nd_gradients
# 3) Functions for the visualizations: model, observations (and derivatives)
from Forward_Tools.Model_Display import *
from Forward_Tools.Observations_Display import *


if __name__ == '__main__':
    # 1) Define general parameters for the model
    # Note: don't define the observation points 'r' here!
    #       The observation points will be defined in the function 'grid_1d_model' and convert to (x,y) coordinate.
    model_general = {
        # Layer related parameters
        'model_flag': 1,  # Different titles for the model image: 1: isotropic model; 2: anisotropic model
        'NL': 8,          # Number of layers
        'iL_pp': 6,       # The index of the reservoir layer, start from 1.
        'H': np.array([50, 422, 251.03, 246.52, 112.67, 70.21, 105.27, 50.3]),
        # Reservoir related parameters
        'R': 900,              # Reservoir's radius, unit [m]
        'P_reservoir': 0.0e6,  # unit [Pa]
        # Geo-mechanical parameters
        'Alpha': np.array([0.95, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]),
        'Nu': np.array([0.35, 0.33, 0.18, 0.25, 0.25, 0.2, 0.22, 0.25]),         # Poisson's ratio
        'G': np.array([0.15, 1.2, 10.48, 5.37, 10.23, 11.16, 10.23, 18]) * 1e9,  # Shear modulus, unit [Pa]
        'RHO': np.array([]),
        'VS': np.array([[], []]),
        'VP': np.array([[], []]),
    }
    # Further define the parameters of the model: don't need to change there.
    # 1.1) Pressure in each layer
    model_general['P'] = np.zeros(model_general['NL'])
    model_general['P'][model_general['iL_pp'] - 1] = model_general['P_reservoir']
    # 1.2) Top and bottom depth of each layer
    model_general['Z_top'] = np.zeros(model_general['NL'])
    model_general['Z_bot'] = model_general['H']

    # 1) Assign the pressure sources to the sub-reservoirs in the target layer (reservoir layer)
    # centers, alphas, betas, amplitudes are introduced in the function 'assign_generalized_gaussian_pressure'
    num_subs = 25                                     # 25 sub-reservoirs in the x or y directions. Total number is 25**2
    centers = [(13, 13), (13, 13), (13, 13)]
    alphas = [(2, 2), (5, 5), (8, 8)]
    betas = [(2.5, 2.5), (2.5, 2.5), (2.5, 2.5)]
    amplitudes = [10.0 * 1e5, 8.0 * 1e5, 6.0 * 1e5]   # max = 2.4e6 [Pa]
    sub_pressures = assign_generalized_gaussian_pressure(num_subs, centers, alphas, betas, amplitudes)

    # 2) Define the auxiliary parameters
    r_max = 2 * model_general['R']   # The maximum radius of the observation point, unit [m]
    dx = 200                         # spacing in x direction = 200
    dy = 200                         # spacing in y direction = 200
    # Print some information, don't need to change there.
    Le = model_general['R'] / num_subs
    Hob = np.sum(model_general['H'][:model_general['iL_pp'] - 1])
    print(f"{num_subs**2} sub-reservoirs are grided. {(ceil(r_max/dx*2)+1) * (ceil(r_max/dy*2)+1)} observation points on the surface.")
    print(f"Le = {Le}, Hob = {Hob}. The Le / Hob = {Le / Hob:.2f} should be less than 1.4.")

    # 3) Observations computation: observations, 1st, and 2nd derivatives
    # 3.1) original observations: uz, ur, tilt_x, tilt_y
    X, Y, uz_total, ur_total, tilt_x_total, tilt_y_total = grid_1d_model(R=model_general['R'], num_subs=num_subs, r_max=r_max, dx=dx, dy=dy,
                                                                         model_general=model_general, sub_pressures=sub_pressures, process_num=14)
    # 3.2) 1st derivatives of uz, ur, tilt_x, tilt_y
    grad_uz_total= compute_1st_gradients(X, Y, uz_total)
    grad_ur_total = compute_1st_gradients(X, Y, ur_total)
    grad_tilt_x_total = compute_1st_gradients(X, Y, tilt_x_total)
    grad_tilt_y_total = compute_1st_gradients(X, Y, tilt_y_total)
    # 3.3) 2nd derivatives of uz, ur, tilt_x, tilt_y
    second_grad_uz_total = compute_2nd_gradients(X, Y, uz_total)
    second_grad_ur_total = compute_2nd_gradients(X, Y, ur_total)
    second_grad_tilt_x_total = compute_2nd_gradients(X, Y, tilt_x_total)
    second_grad_tilt_y_total = compute_2nd_gradients(X, Y, tilt_y_total)

    # 4) Visualization: Call plotting functions
    # 4.1) Vertical profile: All layers
    plot_model_original(model_general=model_general, r_max=r_max, dr=dx,
                        file_name=f"reservoir_model_original_{model_general['NL']}layer.png")
    # 4.2) Horizontal profile: Target layer with sub-reservoirs
    plot_model(R=model_general['R'], num_subs=num_subs, sub_pressures=sub_pressures)
    # 4.3) Plot 4 observations in one figure
    plot_observations(X=X, Y=Y, uz_total=uz_total, ur_total=ur_total, tilt_x_total=tilt_x_total, tilt_y_total=tilt_y_total)
    # 4.4) Options
    # Plot 1st derivatives of 4 observations in one figure
    plot_1st_derivative_observations(X, Y, grad_uz_total, grad_ur_total, grad_tilt_x_total, grad_tilt_y_total)
    # Plot 2nd derivatives of 4 observations in one figure
    plot_2nd_derivative_observations(X, Y, second_grad_uz_total, second_grad_ur_total, second_grad_tilt_x_total, second_grad_tilt_y_total)