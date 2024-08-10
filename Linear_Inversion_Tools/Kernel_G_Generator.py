from math import ceil
import numpy as np
import pickle
import copy
from Forward_Tools.Observations_Computation import *


def kernel_g_generator(model_general, model_prior, auxiliary_parameters, data_type, save=True):
    """
    Compute the kernel matrix G for the linear relationship between the model and data vectors (displacement): d=Gm
    :param model_general: The dict includes all raw information about the multilayer model
    :param model_prior: The maximum likelihood model = model prior
    :param auxiliary_parameters: Auxiliary parameters used in 'grid_1d_model'
    :param data_type: The type of the data vector: 'uz', 'ur', 'tiltx', 'tilty'
    :param save: Whether to save the kernel matrix G
    :return: The kernel matrix G; size = (N, M).
    When using displacement data, the unit is [mm] (will be converted from [m] to [mm]).
    When using tilt data, the unit is [micro-radian].
    """
    # 1) Initialize the G size
    model_size = len(model_prior)
    data_size = (ceil(auxiliary_parameters['r_max']*2/auxiliary_parameters['dx']) + 1) * (ceil(auxiliary_parameters['r_max']*2/auxiliary_parameters['dy']) + 1)
    G = np.zeros((data_size, model_size))
    print(f'The size of Kernel G (for {data_type}) is {data_size}x{model_size}. The Kernel G generation begins:')
    # 2) Loop the model_vector's element to compute the corresponding column of the kernel matrix G
    # Note: only one value is non-zero when looping the model_vector
    for m in range(model_size):
        model_vector_tmp = copy.deepcopy(model_prior)
        model_vector_current = np.zeros_like(model_vector_tmp)
        model_vector_current[m] = model_vector_tmp[m]  # Only one value is non-zero!
        # 2.1) Compute the corresponding data vector d_current
        # Note: all observations are 2D, remember to flatten them to 1D vector
        X, Y, uz_total, ur_total, tilt_x_total, tilt_y_total = grid_1d_model(R=model_general['R'],
                                                                             num_subs=auxiliary_parameters['num_subs'],
                                                                             r_max=auxiliary_parameters['r_max'],
                                                                             dx=auxiliary_parameters['dx'],
                                                                             dy=auxiliary_parameters['dy'],
                                                                             model_general=model_general,
                                                                             sub_pressures=model_vector_current,  # This term updates everytime
                                                                             process_num=auxiliary_parameters['process_num'],
                                                                             print_info=auxiliary_parameters['print_info'])
        # 2.2) Give the current G's column based on the current data vector
        if data_type == 'uz':
            data_vector_current = uz_total.flatten() / model_vector_tmp[m]  # Default unit [m]
        elif data_type == 'ur':
            data_vector_current = ur_total.flatten() / model_vector_tmp[m]  # Default unit [m]
        elif data_type == 'tiltx':
            data_vector_current = tilt_x_total.flatten() / model_vector_tmp[m]    # Default unit [micro-radian]
        elif data_type == 'tilty':
            data_vector_current = tilt_y_total.flatten() / model_vector_tmp[m]    # Default unit [micro-radian]
        else:
            raise ValueError("The data type is not supported. Please use 'uz', 'ur', 'tiltx' and 'tilty'.")
        G[:, m] = data_vector_current
    # Return the G
    print(f"The Kernel G (for {data_type}) generation is successfully completed.")
    if save:
        kernel_g_save(G, data_type)
        print(f"The Kernel G (for {data_type}) is saved.")
    return G


def kernel_g_save(G, data_type):
    """
    Save the kernel matrix G into a .pkl file
    :param G: The kernel matrix G; size = (N, M)
    :param data_type: The type of the data vector: 'uz', 'ur', 'tiltx', 'tilty'
    :return: None
    """
    g_size = G.shape
    filename = f"Kernel_G_{data_type}_{g_size[0]}x{g_size[1]}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
