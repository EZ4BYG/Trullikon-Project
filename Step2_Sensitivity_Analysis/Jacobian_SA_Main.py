import numpy as np
from Jacobian_Matrix_Sensitivity_Analysis import *


if __name__ == '__main__':
    # 1) Define a model with all generally needed parameters
    # Note 1: only 'Z_bot' directly influences the final results, not 'H' or 'Z_top'.
    model_general = {
        # 1.1) parameters don't change: mean values
        'model_flag': 1,
        'NL': 8,
        'NL_real': 8,
        'iL_pp': 6,
        'H': np.array([50, 422, 251.03, 246.52, 112.67, 70.21, 105.27, 50.3]),
        'R': 150.0,
        'Z_top': np.zeros(8),
        'Z_bot': np.array([50, 422, 251.03, 246.52, 112.67, 70.21, 105.27, 50.3]),
        'r': np.arange(1, 16) * 60,
        # 1.2) parameters change for global sensitivity analysis
        'p': np.array([0, 0, 0, 0, 0, 5, 0, 0]) * 1e6,
        'Alpha': np.array([0.95, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]),
        'G': np.array([0.15, 1.2, 10, 5, 10, 11, 10, 18]) * 1e9,
        'nu': np.array([0.35, 0.33, 0.18, 0.25, 0.25, 0.2, 0.22, 0.25]),
        # 1.3) parameters don't use
        'RHO': np.array([]),
        'VS': np.array([]),
        'VP': np.array([]),
    }

    # 1) Define custom parameter ranges for each model parameter
    params_dict = {
        'G': {'layer': [5, 6, 7],
              'range': [[5.92e9, 14.54e9], [7.12e9, 15.2e9], [5.92e9, 14.54e9]],
              'Name': [r'L5 $G$', r'L6 $G$', r'L7 $G$']},
        'nu': {'layer': [5, 6, 7],
               'range': [[0.11, 0.46], [0.14, 0.3], [0.11, 0.46]],
               'Name': [r'L5 $\nu$', r'L6 $\nu$', r'L7 $\nu$']},
        'Alpha': {'layer': [5, 6, 7],
                  'range': [[0.7, 0.9], [0.7, 0.9], [0.7, 0.9]],
                  'Name': [r'L5 $\alpha$', r'L6 $\alpha$', r'L7 $\alpha$']},
        # Note: 'R' doesn't have the 'layer' information
        'R': {'range': [[120, 200]],
              'Name': [r'$R$']},
        # Note: 'p' only has one 'layer' information
        'p': {'layer': [6],
              'range': [[1.2e6, 2.4e6]],
              'Name': [r'$P$']},
        'Z_bot': {'layer': [6],
                  'range': [[50, 100]],
                  'Name': [r'$H$']}
    }
    filename = "SA_Important_Together.png"

    # 2) Consider all 'G'
    # params_dict = {
    #     'G': {'layer': [1, 2, 3, 4, 5, 6, 7],
    #           'range': [[0.08e9, 0.21e9], [1.0e9, 1.4e9], [6.22e9, 14.74e9], [2.82, 7.92],
    #                     [5.92e9, 14.54e9], [7.12e9, 15.2e9], [5.92e9, 14.54e9]],
    #           'Name': [r'L1 $G$', r'L2 $G$', r'L3 $G$', r'L4 $G$',
    #                    r'L5 $G$', r'L6 $G$', r'L7 $G$']}
    # }
    # filename = "SA_All_G.png"

    # 3) Consider all 'Poisson's ratio (nu)'
    # params_dict = {
    #     'nu': {'layer': [1, 2, 3, 4, 5, 6, 7],
    #            'range': [[0.34, 0.40], [0.32, 0.35], [0.11, 0.23], [0.19, 0.41],
    #                      [0.11, 0.46], [0.14, 0.30], [0.11, 0.46]],
    #            'Name': [r'L1 $\nu$', r'L2 $\nu$', r'L3 $\nu$', r'L4 $\nu$',
    #                     r'L5 $\nu$', r'L6 $\nu$', r'L7 $\nu$']}
    # }
    # filename = "SA_All_nu.png"

    # 4) Consider all 'Biot's coefficient (Alpha)'
    # params_dict = {
    #     'nu': {'layer': [1, 2, 3, 4, 5, 6, 7],
    #            'range': [[0.90, 0.97], [0.90, 0.92], [0.7, 0.9], [0.7, 0.9],
    #                      [0.7, 0.9], [0.7, 0.9], [0.7, 0.9]],
    #            'Name': [r'L1 $\alpha$', r'L2 $\alpha$', r'L3 $\alpha$', r'L4 $\alpha$',
    #                     r'L5 $\alpha$', r'L6 $\alpha$', r'L7 $\alpha$']}
    # }
    # filename = "SA_All_alpha.png"

    # Computation the Jacobian matrix
    perturbation = 0.01
    jacobian = compute_jacobian_matrix(model_params=model_general, params_dict=params_dict, obs='tilt', perturbation=perturbation)
    # Get the parameter ranges which are used to standardize the Jacobian matrix and plot the sensitivity matrix
    param_ranges = get_param_ranges(params_dict)
    standardized_jacobian = uncertainty_adjusted_jacobian(jacobian=jacobian, param_ranges=param_ranges)
    # Visualization
    plot_jacobian_matrix(standardized_jacobian, params_dict, model_general['r'],
                         obs='tilt', perturbation=perturbation, filename=filename)


