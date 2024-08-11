import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from Forward_Tools.GeertsmaSol_JP import GeertsmaSol_JP_py


def get_param_ranges(param_group):
    ranges = []
    for key, value in param_group.items():
        for i in range(len(value['range'])):
            range_val = value['range'][i]
            ranges.append(range_val[1] - range_val[0])
    return np.array(ranges)


def uncertainty_adjusted_jacobian(jacobian, param_ranges):
    return jacobian * param_ranges[np.newaxis, :]


def tilt_convert(model_params: dict, uz: np.ndarray):
    uz1 = uz
    r1 = model_params['r']
    r2 = r1 + 1e-3
    [uz2, _] = GeertsmaSol_JP_py(
        NL=model_params['NL'],
        G=model_params['G'],
        nu=model_params['nu'],
        alpha=model_params['Alpha'],
        p=model_params['p'],
        z_top=model_params['Z_top'],
        z_bot=model_params['Z_bot'],
        r=r2,
        R=model_params['R'],
        iL_pp=model_params['iL_pp'],
        RHO=model_params['RHO'],
        VS=model_params['VS'],
        VP=model_params['VP']
    )
    uz2 = np.real(uz2)[0]
    duz = uz2 - uz1
    dr = r2 - r1
    tilt = np.array([np.arctan(-duz[i] / dr[i]) * 1e6 for i in range(len(duz))])
    return tilt


def forward_model(model_params, obs):
    [uz, ur] = GeertsmaSol_JP_py(
        NL=model_params['NL'],
        G=model_params['G'],
        nu=model_params['nu'],
        alpha=model_params['Alpha'],
        p=model_params['p'],
        z_top=model_params['Z_top'],
        z_bot=model_params['Z_bot'],
        r=model_params['r'],
        R=model_params['R'],
        iL_pp=model_params['iL_pp'],
        RHO=model_params['RHO'],
        VS=model_params['VS'],
        VP=model_params['VP']
    )
    uz = np.real(uz)[0]
    ur = np.real(ur)[0]
    tilt = tilt_convert(model_params=model_params, uz=uz)
    if obs == 'uz':
        return uz * 1e3  # Convert to mm
    elif obs == 'ur':
        return ur * 1e3  # Convert to mm
    elif obs == 'tilt':  # Unit: microradian
        return tilt
    else:
        raise ValueError(f"Invalid observation type: {obs}")


def compute_jacobian_matrix(model_params, params_dict, obs='uz', perturbation=0.01):
    n_observations = len(model_params['r'])
    total_params = sum(len(value['layer']) if 'layer' in value else 1 for value in params_dict.values())
    jacobian = np.zeros((n_observations, total_params))

    # 'base_params' records the base model parameters
    base_params = deepcopy(model_params)

    # Compute the Jacobian matrix by a small perturbation
    # Outer loop through each parameter: inner loop 'layer' to compute the sensitivity
    idx = 0
    for param, value in params_dict.items():
        if 'layer' in value:
            for layer in value['layer']:
                # 'perturbed_params' records the perturbed model parameters based on the base model parameters
                perturbed_params = deepcopy(base_params)
                # Get the new parameter value at the current 'layer'
                original_value = base_params[param][layer - 1]
                perturbation_value = original_value * perturbation
                new_value = original_value + perturbation_value
                perturbed_params[param][layer - 1] = new_value
                # Compute the old and new observation (using the small perturbation)
                observations_perturbed = forward_model(model_params=perturbed_params, obs=obs)  # use new value
                observations_original = forward_model(model_params=base_params, obs=obs)        # use original value
                # Compute the sensitivity: (d_obs_perturbed - d_obs_original) / perturbation
                jacobian[:, idx] = (observations_perturbed - observations_original) / perturbation_value
                idx += 1
                # Don't forget to reset the perturbed parameter value back to the original value !!!
                perturbed_params[param][layer - 1] = original_value
        # Specific case: 'R' doesn't have the 'layer' information
        else:
            # 'perturbed_params' records the perturbed model parameters based on the base model parameters
            perturbed_params = deepcopy(base_params)
            # Get the new parameter value
            original_value = base_params[param]
            perturbation_value = original_value * perturbation
            new_value = original_value + perturbation_value
            perturbed_params[param] = new_value
            # Compute the old and new observation (using the small perturbation)
            observations_perturbed = forward_model(model_params=perturbed_params, obs=obs)
            observations_original = forward_model(model_params=base_params, obs=obs)
            # Compute the sensitivity: (d_obs_perturbed - d_obs_original) / perturbation
            jacobian[:, idx] = (observations_perturbed - observations_original) / perturbation_value
            idx += 1

    # Finish the computation
    print('Jacobian matrix computed successfully.')
    return jacobian


###########################################################################
#######################          Visualization     ########################
###########################################################################
def plot_jacobian_matrix(jacobian_matrix, params_dict, observation_points, perturbation,
                         obs='uz', filename='standard_jacobian_matrix.png'):
    """
    Visualize the Jacobian matrix as a heatmap.

    :param jacobian_matrix: The Jacobian matrix to visualize.
    :param params_dict: Dictionary specifying which parameters and layers to consider for sensitivity analysis.
    :param observation_points: List of observation points.
    :param perturbation: Perturbation size used in Jacobian matrix computation.
    :param obs: Type of observation data used ('uz', 'ur', 'tilt').
    :param filename: Filename to save the heatmap.
    """
    # Define the labels for the x and y axes
    x_labels = []
    for param, value in params_dict.items():
        for i in range(len(value['Name'])):
            x_labels.append(value['Name'][i])
    y_labels = observation_points

    # Define the image and colorbar titles based on the observation type
    if obs == 'uz':
        title = 'Uncertainty-Adjusted Jacobian Matrix Sensitivity Analysis\n Observation Data (Vertical Displacement)'
        colorbar_title = 'Sensitivity (uz [mm])'
    elif obs == 'ur':
        title = 'Uncertainty-Adjusted Jacobian Matrix Sensitivity Analysis\n Observation Data (Horizontal Displacement)'
        colorbar_title = 'Sensitivity (ur [mm])'
    elif obs == 'tilt':
        title = 'Uncertainty-Adjusted Jacobian Matrix Sensitivity Analysis\n Observation Data (Tilt)'
        colorbar_title = 'Sensitivity (tilt [microradian])'
    else:
        raise ValueError("Invalid observation type. Choose from 'uz', 'ur', or 'tilt'.")
    title += f' with {perturbation * 100}% Model Parameter Perturbation'

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(jacobian_matrix, annot=True, fmt=".3f", cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels)
    plt.xlabel('Model Parameters', fontsize=12)
    plt.ylabel('Observation Points [m]', fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_title, fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
