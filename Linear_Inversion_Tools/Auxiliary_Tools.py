import numpy as np


def add_gaussian_noise_to_data(data, data_noise_standard_deviation):
    """
    Add a Gaussian noise (mean=0, so bias=0; deviation represents the variance) to every data point.
    :param data: The data matrix (2d ndarray) or vector (1d ndarray)
    :param data_noise_standard_deviation: The standard deviation of the Gaussian noise (uncertainty), it's an absolute value. The unit is the same as the data.
    :return: The noise data with the same shape as the input data
    """
    noise = np.random.normal(0, data_noise_standard_deviation, size=data.shape)
    data_noise = data + noise
    return data_noise


def add_gaussian_noise_to_model(model, model_noise_standard_deviation):
    """
    Add a Gaussian noise (mean=0, so bias=0; deviation represents the variance) to every model point.
    :param model: The model matrix (2d ndarray) or vector (1d ndarray)
    :param model_noise_standard_deviation: The standard deviation of the Gaussian noise (uncertainty), it's an absolute value. The unit is the same as the model!
    :return: The noise model with the same shape as the input model
    """
    noise = np.random.normal(0, model_noise_standard_deviation, size=model.shape)
    model_noise = model + noise
    return model_noise


def root_mean_square_misfit(dobs, dpred, data_noise_standard_deviation):
    """
    Compute the root-mean-square error (RMSE) between the observed data and the predicted data.
    :param dobs: The observed data vector (with noise); size = (N, 1)
    :param dpred: The estimated data vector; size = (N, 1)
    :param data_noise_standard_deviation: The standard deviation of the prior data noise
    :return: The root-mean-square error (RMSE), a value.
    """
    RMSE = np.sqrt(1 / dobs.size * np.sum((dobs - dpred) ** 2))
    print("The root-mean-square error (RMSE) between the observed data and the predicted data is: ", RMSE)
    if RMSE < 0.5*data_noise_standard_deviation:
        print(f"The RMSE is smaller than the prior data noise standard deviation {data_noise_standard_deviation}: Over-fitting.")
    elif RMSE > 2*data_noise_standard_deviation:
        print(f"The RMSE is larger than the prior data noise standard deviation {data_noise_standard_deviation}: Under-fitting.")
    else:
        print(f"The RMSE approximates the prior data noise standard deviation {data_noise_standard_deviation}: Good-fitting.")
    return RMSE


def standardized_root_mean_square_misfit(dobs, dpred, Cd, data_noise_standard_deviation):
    """
    Compute the standardized root-mean-square error (RMSE): Formula (6.25) in Andreas Fichtner's Book, 2021.
    :param dobs: The observed data vector (with noise); size = (N, 1)
    :param dpred: The estimated data vector; size = (N, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param data_noise_standard_deviation: The standard deviation of the prior data noise
    :return: The standardized root-mean-square error (RMSE), a value.
    """
    N = len(dobs)
    SRMSE = np.sqrt(1 / N * np.matmul(np.matmul((dobs - dpred).T, np.linalg.inv(Cd)), (dobs - dpred)))
    print("The standardized root-mean-square error (SRMSE) between the observed data and the predicted data is: ", SRMSE)
    if SRMSE < 0.5:
        print(f"The SRMSE is smaller than 0.5: Over-fitting.")
    elif SRMSE > 2:
        print(f"The SRMSE is larger than 2: Under-fitting.")
    else:
        print(f"The SRMSE approximates 1: Good-fitting.")
    return SRMSE


def matrix_vector_concatenate(*arrays, axis=0, save=True):
    """
    Concatenate the input arrays (matrix or vector) along the specified axis.
    :param arrays: The input arrays to be concatenated
    :param axis: The axis along which the arrays will be concatenated
    :param save: Whether to save the concatenated matrix / vector
    :return: The concatenated matrix / vector
    """
    # Check that all other dimensions except the concatenation axis are the same
    other_dims = [tuple(arr.shape[i] for i in range(arr.ndim) if i != axis) for arr in arrays]
    if not all(dim == other_dims[0] for dim in other_dims):
        raise ValueError("All arrays must have the same dimensions in non-concatenation axes.")

    # Concatenate the arrays
    joint = np.concatenate(arrays, axis=axis)
    if save:
        np.save(f"Joint_axis={axis}.npy", joint)
    return joint


def matrix_diagonal_stack(*matrices):
    """
    Stack an arbitrary number of 2D matrices diagonally.
    :param matrices: The input matrices to be stacked
    :return: The stacked matrix
    """
    # 1) Total size calculation along each dimension
    total_size = sum([m.shape[0] for m in matrices])
    # 2) Initialize a large zero matrix to hold all the input matrices
    result = np.zeros((total_size, total_size))
    # 3) Position to start inserting each matrix
    current_position = 0
    # 4) Loop through each matrix and place it in the correct diagonal position
    for matrix in matrices:
        size = matrix.shape[0]
        result[current_position:current_position + size, current_position:current_position + size] = matrix
        current_position += size
    return result



























    return SRMSE