import numpy as np


def generate_steepness_matrix(M, delta_m):
    """
    Generate the steepness matrix (1st derivative) Wm. The size is MxM.
    :param M: The number of the model parameters
    :param delta_m: The spacing between the model parameters
    :return: The steepness matrix Wm; size = (M, M)
    """
    D = np.zeros((M, M))
    for i in range(M - 1):
        D[i, i] = -1 / delta_m
        D[i, i + 1] = 1 / delta_m
    # Compute the steepness matrix Wm; size = (M, M)
    Wm = np.matmul(D.T, D)
    return Wm


def generate_roughness_matrix(M, delta_m):
    """
    Generate the roughness matrix (2nd derivative) Wm. The size is MxM.
    :param M: The number of the model parameters
    :param delta_m: The spacing between the model parameters
    :return: The roughness matrix Wm; size = (M, M)
    """
    D = np.zeros((M - 2, M))
    for i in range(M - 2):
        D[i, i] = 1 / delta_m**2
        D[i, i + 1] = -2 / delta_m**2
        D[i, i + 2] = 1 / delta_m**2
    # Compute the roughness matrix Wm; size = (M, M)
    Wm = np.matmul(D.T, D)
    return Wm


#####################################################################################
############  Bayesian Least Square Solution: Prior covariance matrix  ##############
#####################################################################################
def generate_prior_data_covariance_matrix(data_size, data_noise_standard_deviation):
    """
    Generate the data covariance matrix Cd (or We). The size is NxN.
    :param data_size: The size of the data vector (N)
    :param data_noise_standard_deviation: same as the input of add_gaussian_noise_to_data()
    :return: The data covariance matrix Cd; size = (N, N)
    """
    prior_data_covariance_matrix = np.eye(data_size) * (data_noise_standard_deviation**2)
    return prior_data_covariance_matrix


def generate_prior_model_covariance_matrix(model_size, model_uncertainty_standard_deviation):
    """
    Generate the model covariance matrix Cm (or Wm). The size is MxM.
    :param model_size: The size of the model vector (M)
    :param model_uncertainty_standard_deviation: same as the input of add_gaussian_noise_to_model()
    :return: The model covariance matrix Cm; size = (M, M)
    """
    prior_model_covariance_matrix = np.eye(model_size) * (model_uncertainty_standard_deviation**2)
    return prior_model_covariance_matrix


def generate_gaussian_kernel_prior_model_covariance_matrix(model_size, model_uncertainty_standard_deviation, model_lambda):
    """
    Generate the Gaussian kernel prior model covariance matrix Cm (or Wm). The size is MxM.
    :param model_size: The size of the model vector (M)
    :param model_uncertainty_standard_deviation: same as the input of add_gaussian_noise_to_model()
    :param model_lambda: The length scale of the Gaussian kernel, a hyperparameter.
    :return: The Gaussian kernel prior model covariance matrix Cm; size = (M, M)
    """
    Cm = np.zeros((model_size, model_size))
    for i in range(model_size):
        for j in range(model_size):
            distance = abs(i - j)
            Cm[i, j] = model_uncertainty_standard_deviation**2 * np.exp(-distance**2 / model_lambda**2)
    return Cm


#####################################################################################
#########  Bayesian Least Square Solution: Model and Data resolution matrix  ########
#####################################################################################
def generate_model_resolution_matrix(G, Cd, Cm):
    """
    Generate the model resolution matrix RM of the Bayesian least square solution. RM = Gg * G. The size is MxM.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :return: The model resolution matrix RM; size = (M, M)
    """
    # 1) Prepare some terms
    Cm_Gt = np.matmul(Cm, G.T)
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    Cd_G_Cm_Gt = Cd + G_Cm_Gt
    # 2) Don't compute the inverse of Cd_G_Cm_Gt directly, we can solve the linear system
    last_2_term = np.linalg.solve(Cd_G_Cm_Gt, G)
    # 3) Compute the RM
    RM = np.matmul(Cm_Gt, last_2_term)
    return RM


def generate_data_resolution_matrix(G, Cd, Cm):
    """
    Generate the data resolution matrix RD of the Bayesian least square solution. RD = G * Gg. The size is NxN.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :return: The data resolution matrix RD; size = (N, N)
    """
    # 1) Compute the general kernel Gg
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    Cd_G_Cm_Gt = Cd + G_Cm_Gt
    Cd_G_Cm_Gt_inverse = np.linalg.inv(Cd_G_Cm_Gt)
    Gg = np.matmul(np.matmul(Cm, G.T), Cd_G_Cm_Gt_inverse)
    # 2) Compute the RD
    RD = np.matmul(G, Gg)
    return RD


##################################################################################################################
#########  Bayesian Least Square Solution: Model and Data resolution matrix with Tikhonov regularization  ########
#########  1) We add the regularization term to the prior data covariance matrix Cd                       ########
##################################################################################################################
def generate_model_resolution_matrix_tikhonov_regularization_cd(G, Cd, Cm, lambda_reg):
    """
    Generate the model resolution matrix RM of the Bayesian least square solution with Tikhonov regularization.
    RM = Gg * G. The size is MxM.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The model resolution matrix RM; size = (M, M)
    """
    # 1) Compute G * Cm * G^T and the regularization term lambda * I
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    reg_term = lambda_reg * np.eye(G_Cm_Gt.shape[0])
    # Compute the modified matrix Cd + G * Cm * G^T + lambda * I
    Cd_G_Cm_Gt_reg = Cd + G_Cm_Gt + reg_term
    # Compute the inverse of the modified matrix
    Cd_G_Cm_Gt_reg_inverse = np.linalg.inv(Cd_G_Cm_Gt_reg)
    # 2) Compute the general kernel Gg
    Gg = np.matmul(np.matmul(Cm, G.T), Cd_G_Cm_Gt_reg_inverse)
    # 3) Compute the RM
    RM = np.matmul(Gg, G)
    return RM


def generate_data_resolution_matrix_tikhonov_regularization_cd(G, Cd, Cm, lambda_reg):
    """
    Generate the data resolution matrix RD of the Bayesian least square solution with Tikhonov regularization.
    RD = G * Gg. The size is NxN.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The data resolution matrix RD; size = (N, N)
    """
    # Compute G * Cm * G^T and the regularization term lambda * I
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    reg_term = lambda_reg * np.eye(G_Cm_Gt.shape[0])
    # Compute the modified matrix Cd + G * Cm * G^T + lambda * I
    Cd_G_Cm_Gt_reg = Cd + G_Cm_Gt + reg_term
    # Compute the inverse of the modified matrix
    Cd_G_Cm_Gt_reg_inverse = np.linalg.inv(Cd_G_Cm_Gt_reg)
    # Compute the general kernel Gg
    Gg = np.matmul(np.matmul(Cm, G.T), Cd_G_Cm_Gt_reg_inverse)
    # Compute the RD
    RD = np.matmul(G, Gg)
    return RD


##################################################################################################################
#########  Bayesian Least Square Solution: Model and Data resolution matrix with Tikhonov regularization  ########
#########  2) We add the regularization term to the prior model covariance matrix Cm                      ########
##################################################################################################################
def generate_model_resolution_matrix_tikhonov_regularization_cm(G, Cd, Cm, lambda_reg):
    """
    Generate the model resolution matrix RM of the Bayesian least square solution with Tikhonov regularization.
    RM = Gg * G. The size is MxM.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The model resolution matrix RM; size = (M, M)
    """
    # 1) Add the regularization term lambda * I to Cm
    Cm_reg = Cm + lambda_reg * np.eye(Cm.shape[0])
    # 2) Compute G * Cm_reg * G^T
    G_Cm_reg_Gt = np.matmul(np.matmul(G, Cm_reg), G.T)
    # 3) Compute the modified matrix Cd + G * Cm_reg * G^T
    Cd_G_Cm_reg_Gt = Cd + G_Cm_reg_Gt
    Cd_G_Cm_reg_Gt_inverse = np.linalg.inv(Cd_G_Cm_reg_Gt)
    # 4) Compute the general kernel Gg
    Gg = np.matmul(np.matmul(Cm_reg, G.T), Cd_G_Cm_reg_Gt_inverse)
    # 5) Compute the RM
    RM = np.matmul(Gg, G)
    return RM


def generate_data_resolution_matrix_tikhonov_regularization_cm(G, Cd, Cm, lambda_reg):
    """
    Generate the data resolution matrix RD of the Bayesian least square solution with Tikhonov regularization.
    RD = G * Gg. The size is NxN.
    :param G: The kernel matrix; size = (N, M)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The data resolution matrix RD; size = (N, N)
    """
    # 1) Add the regularization term lambda * I to Cm
    Cm_reg = Cm + lambda_reg * np.eye(Cm.shape[0])
    # 2) Compute G * Cm_reg * G^T
    G_Cm_reg_Gt = np.matmul(np.matmul(G, Cm_reg), G.T)
    # 3) Compute the modified matrix Cd + G * Cm_reg * G^T
    Cd_G_Cm_reg_Gt = Cd + G_Cm_reg_Gt
    Cd_G_Cm_reg_Gt_inverse = np.linalg.inv(Cd_G_Cm_reg_Gt)
    # 4) Compute the general kernel Gg
    Gg = np.matmul(np.matmul(Cm_reg, G.T), Cd_G_Cm_reg_Gt_inverse)
    # 5) Compute the RD
    RD = np.matmul(G, Gg)
    return RD
