import numpy as np


def adjust_model_for_observation_depth(model_general, z_obs):
    """
    Adjust the model parameters in model_general based on the observation depth z_obs.
    :param model_general: Original model_general dictionary containing the model parameters
    :param z_obs: Observation depth, a negative value representing depth below the surface
    :return: Updated model_general dictionary with adjusted parameters
    """
    # 1) Check if the observation depth is valid
    # First check: Ensure z_obs is negative
    if z_obs >= 0:
        raise ValueError("Observation depth z_obs must be a negative value.")
    # Second check: Ensure the observation depth is not deeper than the reservoir layer
    if abs(z_obs) > sum(model_general['H'][:model_general['iL_pp']-1]):
        raise ValueError("Observation depth z_obs cannot be deeper than the reservoir layer depth.")

    # 2) Find the index of the first layer to keep and adjust the thickness of the first layer if necessary
    cumulative_depth = 0
    first_layer_index = 0
    for i, thickness in enumerate(model_general['H']):
        if cumulative_depth + thickness > abs(z_obs):
            break
        cumulative_depth += thickness
        first_layer_index = i + 1

    # 3) Adjust the thickness of the first layer if necessary
    if first_layer_index < len(model_general['H']):
        model_general['H'][first_layer_index] -= (abs(z_obs) - cumulative_depth)

    # 4) Adjust other model_general parameters
    model_general['NL'] -= first_layer_index
    model_general['NL_real'] -= first_layer_index
    model_general['iL_pp'] -= first_layer_index
    model_general['H'] = model_general['H'][first_layer_index:]
    model_general['Alpha'] = model_general['Alpha'][first_layer_index:]
    model_general['Nu'] = model_general['Nu'][first_layer_index:]
    model_general['G'] = model_general['G'][first_layer_index:]

    return model_general
