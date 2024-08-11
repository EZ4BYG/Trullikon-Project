import time
import numpy as np
# 1) For forward modeling
from Forward_Tools.Model_Display import plot_model
from Forward_Tools.Gaussian_Pressure_Source import *
from Forward_Tools.Generalized_Gaussian_Pressure_Source import *
# 2) For linear inversion
from Linear_Inversion_Tools.Linear_Inversion_Algorithms import *
from Linear_Inversion_Tools.Kernel_G_Generator_Faster import kernel_g_generator_faster
from Linear_Inversion_Tools.Important_Matrix_Generator import *
from Linear_Inversion_Tools.Auxiliary_Tools import *
from Linear_Inversion_Tools.Results_Display import *
from Linear_Inversion_Tools.Important_Matrix_Display import *
# 3) For posterior sampling
from Linear_Inversion_Tools.Posterior_Sampling import posterior_sampling
from Linear_Inversion_Tools.Posterior_Sampling import pca_transformation, select_samples_within_custom_ranges
from Linear_Inversion_Tools.Posterior_Sampling import plot_pca_samples_with_ellipses, plot_all_selected_models


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
        'dx': 100,
        'dy': 100,
        'process_num': 14,  # Change it according to the computer's configuration
        'print_info': False
    }

    # 3) Generate all Kernel G for the linear inversion: d=Gm
    _, _, G_tiltx, G_tilty = kernel_g_generator_faster(model_general, auxiliary_parameters=auxiliary_parameters_tilt, save=True)
    G_uz, G_ur, _, _ = kernel_g_generator_faster(model_general, auxiliary_parameters=auxiliary_parameters_disp, save=True)
    G_joint = matrix_vector_concatenate(G_tiltx, G_tilty, G_uz, G_ur, axis=0, save=False)

    # 4) Import the true model and generate the data_truth_joint
    truth_model = np.load("model_saved.npy")
    truth_model_flatten = truth_model.flatten()
    plot_model(R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], sub_pressures=truth_model_flatten, file_name="truth_model.png")
    # Generate the true data for each observation type
    data_truth_tiltx = G_tiltx @ truth_model_flatten
    data_truth_tilty = G_tilty @ truth_model_flatten
    data_truth_uz = G_uz @ truth_model_flatten
    data_truth_ur = G_ur @ truth_model_flatten
    # Joint into the same data_truth
    data_truth_joint = matrix_vector_concatenate(data_truth_tiltx, data_truth_tilty, data_truth_uz, data_truth_ur, axis=0, save=False)

    # 5) Generate the synthetic data: data_obs_joint (data_truth_joint + noise)
    tiltx_noise_standard_deviation = 0.03  # Unit: micro-radian
    tilty_noise_standard_deviation = 0.03  # Unit: micro-radian
    uz_noise_standard_deviation = 5        # Unit: mm
    ur_noise_standard_deviation = 3        # Unit: mm
    # Generate the observed data for each observation type
    data_obs_tiltx = add_gaussian_noise_to_data(data=data_truth_tiltx, data_noise_standard_deviation=tiltx_noise_standard_deviation)
    data_obs_tilty = add_gaussian_noise_to_data(data=data_truth_tilty, data_noise_standard_deviation=tilty_noise_standard_deviation)
    data_obs_uz = add_gaussian_noise_to_data(data=data_truth_uz, data_noise_standard_deviation=uz_noise_standard_deviation)
    data_obs_ur = add_gaussian_noise_to_data(data=data_truth_ur, data_noise_standard_deviation=ur_noise_standard_deviation)
    # Joint into the same data_obs
    data_obs_joint = matrix_vector_concatenate(data_obs_tiltx, data_obs_tilty, data_obs_uz, data_obs_ur, axis=0, save=False)

    # 6) Generate and visualize the prior model: using a Gaussian function to quickly generate
    # Option 1: Gaussian prior
    centers = [(12, 14)]
    sigmas = [(5, 5)]
    amplitudes = [3e6]  # Different amplitudes for different regions: max = 8Mpa
    model_prior = np.array(assign_gaussian_pressure(num_subs=auxiliary_parameters_tilt['num_subs'],
                                                    centers=centers, sigmas=sigmas, amplitudes=amplitudes))
    # Option 2: Uniform prior
    # model_prior = np.full(G_joint.shape[1], 3e6)
    plot_prior_model(R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], prior_model=model_prior, file_name=None)

    # 7) Generate the prior uncertainty information: data + model covariance matrix
    # Cd: joint --> Careful of Cd_joint !!! size = (N, N) -> N is the total number of data !!!
    Cd_tiltx = generate_prior_data_covariance_matrix(data_size=len(data_obs_tiltx), data_noise_standard_deviation=tiltx_noise_standard_deviation)
    Cd_tilty = generate_prior_data_covariance_matrix(data_size=len(data_obs_tilty), data_noise_standard_deviation=tilty_noise_standard_deviation)
    Cd_uz = generate_prior_data_covariance_matrix(data_size=len(data_obs_uz), data_noise_standard_deviation=uz_noise_standard_deviation)
    Cd_ur = generate_prior_data_covariance_matrix(data_size=len(data_obs_ur), data_noise_standard_deviation=ur_noise_standard_deviation)
    Cd_joint = matrix_diagonal_stack(Cd_tiltx, Cd_tilty, Cd_uz, Cd_ur)
    # Cm: doesn't joint
    model_uncertainty_standard_deviation = 2e6  # Unit: Pa (~10% of the maximum pressure value in the prior model)
    Cm = generate_prior_model_covariance_matrix(model_size=len(model_prior), model_uncertainty_standard_deviation=model_uncertainty_standard_deviation)

    # 8) Bayesian least-square inversion: joint G, dobs, and Cd
    [posterior_model_mean, posterior_model_covariance] = bayesian_least_square_solution_formula2(G=G_joint, dobs=data_obs_joint, Cd=Cd_joint, model_prior=model_prior, Cm=Cm)
    # Visualization 1: Posterior model mean
    plot_posterior_model(R=model_general['R'], num_subs=auxiliary_parameters_tilt['num_subs'], posterior_model=posterior_model_mean,
                         title="Joint Data Posterior Maximum Likelihood Reservoir Model: Mean Values", file_name="joint_data_posterior_mean_model.png")
