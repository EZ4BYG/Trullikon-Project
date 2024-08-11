import time
import numpy as np
from scipy.optimize import nnls
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
from Linear_Inversion_Tools.Nonnegative_MH_Posterior_Sampling import metropolis_hastings_with_nonnegative_projection
from Linear_Inversion_Tools.Nonnegative_MH_Posterior_Sampling import amha_posterior_analysis_chi, amha_posterior_analysis_distance
from Linear_Inversion_Tools.Nonnegative_MH_Posterior_Sampling import plot_all_selected_mh_models


if __name__ == '__main__':
    ###### Model Preparation ######
    # 1) Define the general parameters for the model
    model_general = {
        # Layer related parameters
        'model_flag': 1,  # Different titles for the model image: 1: isotropic model; 2: anisotropic model
        'NL': 8,
        'NL_real': 8,     # Generally it equals to NL
        'iL_pp': 6,       # The index of the reservoir layer
        'H': np.array([50, 422, 251.03, 246.52, 112.67, 70.21, 105.27, 50.3]),
        # Reservoir related parameters
        'R': 900,         # Reservoir's radius, unit [m]
        'P_reservoir': 2.4e6,
        # Geo-mechanical parameters
        'Alpha': np.array([0.95, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]),     # Biot's coefficient
        'Nu': np.array([0.35, 0.33, 0.18, 0.25, 0.25, 0.2, 0.22, 0.25]),  # Poisson's ratio
        'G': np.array([0.1, 1.2, 10, 5, 11, 13, 11, 18]) * 1e9,           # Shear modulus, unit: Pa
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

    # 2) Define the auxiliary parameters
    auxiliary_parameters = {
        'num_subs': 25,                   # The number of grids in each dimension (x and y)
        'r_max': 2 * model_general['R'],  # The maximum radius of observation point
        'dx': 400,                        # Spacing in x-direction between observation points
        'dy': 400,                        # Spacing in y-direction between observation points
        'process_num': 14,                # 14 processes are used for the parallel computing
        'print_info': False
    }

    # 3) Generate the Kernel G for the linear inversion: d=Gm
    # Note: here we only use 'tiltx' and 'tilty' components for the inversion
    _, _, G_tiltx, G_tilty = kernel_g_generator_faster(model_general, auxiliary_parameters, save=True)
    G_joint_tilt = matrix_vector_concatenate(G_tiltx, G_tilty, axis=0, save=False)

    # 4) Import the true model and generate the true observations ('data_truth_joint_tilt') and noisy observations ('data_obs_joint_tilt')
    truth_model = np.load("model_saved.npy")
    truth_model_flatten = truth_model.flatten()
    # 4.1) Generate the joint true synthetic data:
    data_truth_tiltx = G_tiltx @ truth_model_flatten
    data_truth_tilty = G_tilty @ truth_model_flatten
    data_truth_joint_tilt = matrix_vector_concatenate(data_truth_tiltx, data_truth_tilty, axis=0, save=False)
    # 4.2) Generate the joint noisy synthetic data:
    tiltx_noise_standard_deviation = 0.05  # Unit: micro-radian
    tilty_noise_standard_deviation = 0.03  # Unit: micro-radian
    data_obs_tiltx = add_gaussian_noise_to_data(data=data_truth_tiltx, data_noise_standard_deviation=tiltx_noise_standard_deviation)
    data_obs_tilty = add_gaussian_noise_to_data(data=data_truth_tilty, data_noise_standard_deviation=tilty_noise_standard_deviation)
    data_obs_joint_tilt = matrix_vector_concatenate(data_obs_tiltx, data_obs_tilty, axis=0, save=False)

    # 5) Generate and visualize the prior model: using a Gaussian function to quickly generate
    # Option 1: Using a single Gaussian function
    centers = [(12, 12), (12, 12)]
    sigmas = [(6, 6)]
    amplitudes = [2.4e6]  # Different amplitudes for different regions: max = 8Mpa
    model_prior = np.array(assign_gaussian_pressure(num_subs=auxiliary_parameters['num_subs'], centers=centers, sigmas=sigmas, amplitudes=amplitudes))
    # Option 2: Using a uniform distribution
    # model_prior = np.full(G.shape[1], 2.4e6)

    # 6) Generate the prior uncertainty information: data + model covariance matrix
    # 6.1) Cd_joint_tilt
    Cd_tiltx = generate_prior_data_covariance_matrix(data_size=len(data_obs_tiltx), data_noise_standard_deviation=tiltx_noise_standard_deviation)
    Cd_tilty = generate_prior_data_covariance_matrix(data_size=len(data_obs_tilty), data_noise_standard_deviation=tilty_noise_standard_deviation)
    Cd_joint_tilt = matrix_diagonal_stack(Cd_tiltx, Cd_tilty)
    # 6.2) Cm: don't need joint
    model_uncertainty_standard_deviation = 2e6  # Unit: Pa
    Cm = generate_prior_model_covariance_matrix(model_size=len(model_prior), model_uncertainty_standard_deviation=model_uncertainty_standard_deviation)

    # 7) BNNLSM solution and visualization
    [posterior_model_mean, posterior_model_covariance] = bayesian_non_negative_least_square_solution(G=G_joint_tilt, dobs=data_obs_joint_tilt, Cd=Cd_joint_tilt,
                                                                                                     model_prior=model_prior, Cm=Cm)
    # Visualization 1: Posterior model mean
    plot_posterior_model(R=model_general['R'], num_subs=auxiliary_parameters['num_subs'], posterior_model=posterior_model_mean,
                         title="Bayesian Non-negative Least-Squares Solution", file_name="BNNLSM_result.png")

    ###########################################################################
    ############# Optional 1: Posterior Sampling and Visualization   ##########
    ############# Method 1: Metropolis-Hastings + Non-projection  √  ##########
    ############# Method 2: HMC + Non-projection                     ##########
    ###########################################################################
    num_samples = 20000
    max_value = 3e6
    # 9.1) Posterior model sampling us Adaptive Metropolis-Hastings algorithm with Non-negative projection
    # initial_sample = posterior_model_mean
    # posterior_model_samples = metropolis_hastings_with_nonnegative_projection(initial_sample=initial_sample, num_samples=num_samples, max_value=max_value,
    #                                                                           initial_proposal_std=model_uncertainty_standard_deviation/1e1, model_prior=model_prior,
    #                                                                           Cm=Cm, Cd=Cd_joint_tilt, G=G_joint_tilt, dobs=data_obs_joint_tilt)
    # print("The number of accepted posterior samples: ", posterior_model_samples.shape[0])
    # np.save(f"{num_samples}_BNNLSM_AMHA_posterior_samples.npy", posterior_model_samples)
    posterior_model_samples = np.load(f"{num_samples}_BNNLSM_AMHA_posterior_samples.npy")
    sorted_posterior_chi_samples = amha_posterior_analysis_chi(amha_accepted_samples=posterior_model_samples, initial_sample=posterior_model_mean, bins=50,
                                                               model_prior=model_prior, Cd=Cd_joint_tilt, Cm=Cm, G=G_joint_tilt, dobs=data_obs_joint_tilt,
                                                               trim=30, file_name="amha_posterior_chi_values_distribution.png")
    sorted_posterior_dist_samples = amha_posterior_analysis_distance(amha_accepted_samples=posterior_model_samples, initial_sample=posterior_model_mean, bins=50,
                                                                     trim=30, file_name="amha_posterior_distances_distributions.png")

    # 9.2) Visualization of the selected models: first 9 samples
    first_nums = np.arange(0, 9)
    plot_all_selected_mh_models(R=model_general['R'], num_subs=auxiliary_parameters['num_subs'],
                                samples=sorted_posterior_dist_samples, selected_indices=first_nums, num_cols=3,
                                file_name="BNNLSM_AMHA_posterior_samples.png")

