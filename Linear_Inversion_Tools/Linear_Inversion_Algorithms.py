import numpy as np
from Forward_Tools.Observations_Computation import *
from scipy.optimize import nnls, lsq_linear


###################################################################################################
##### Gm=d is completely over-determined, only has minimum error solution (no exact solution) #####
###################################################################################################
def least_square_solution(G, dobs):
    """
    Compute the least-square solution for the linear inversion: mest = (G^T G)^(-1) G^T dobs
    When use this method, the problem should be completely over-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :return: The least-square solution m; size = (M, 1)
    """
    # 1) Compute the general kernel Gg
    GtG = np.matmul(G.T, G)
    GtG_condition_number = np.linalg.cond(GtG)
    print(f"The condition number of matrix (G^T G) is {GtG_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is inaccurate (unstable).)")
    GtG_inverse = np.linalg.inv(GtG)
    Gg = np.matmul(GtG_inverse, G.T)
    # 2) Compute the least-square solution
    mest = np.matmul(Gg, dobs)
    print(f"Least square solution is successfully computed.")
    return mest


def weighted_least_square_solution(G, dobs, We):
    """
    Compute the weighted least-square solution for the linear inversion: mest = (G^T We G)^(-1) G^T We dobs
    When use this method, the problem should be completely over-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param We: The weight matrix for the data vector d; size = (N, N)
    :return: The weighted least-square solution m; size = (M, 1)
    """
    # 1) Compute the general kernel Gg
    GtWe = np.matmul(G.T, We)
    Gg = np.matmul(GtWe, G)
    Gg_condition_number = np.linalg.cond(Gg)
    print(f"The condition number of matrix (G^T We G) is {Gg_condition_number} \n"
            f"(Note: If the condition number is large (>e12), the inverse of this matrix is inaccurate (unstable).)")
    Gg_inverse = np.linalg.inv(Gg)
    # 2) Compute the weighted least-square solution
    mest = np.matmul(np.matmul(Gg_inverse, GtWe), dobs)
    print(f"Weighted least square solution is successfully computed.")
    return mest


##################################################################################################
##### Gm=d is purely under-determined, the solution is not unique (infinite exact solutions) #####
##################################################################################################
def minimum_length_solution(G, dobs):
    """
    Compute the minimum length solution for the linear inversion: mest = argmin m^T m
    When use this method, the problem should be purely under-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :return: The minimum length solution m; size = (M, 1)
    """
    # 1) Compute the general kernel Gg
    GGt = np.matmul(G, G.T)
    GGt_condition_number = np.linalg.cond(GGt)
    print(f"The condition number of matrix (GG^T) is {GGt_condition_number}. \n"
          f"(Note: When the condition number is large (>e12), the inverse of this matrix is inaccurate (unstable).)")
    GGt_inverse = np.linalg.inv(GGt)
    Gg = np.matmul(G.T, GGt_inverse)
    # 2) Compute the minimum length solution
    mest = np.matmul(Gg, dobs)
    print(f"Minimum length solution is successfully computed.")
    return mest


def weighted_minimum_length_solution(G, dobs, Wm, model_prior):
    """
    Compute the weighted minimum length solution for the linear inversion: mest = argmin (m-m_prior)^T Wm (m-m_prior)
    When use this method, the problem should be purely under-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param Wm: The weight matrix for the model vector m; size = (M, M)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :return: The weighted minimum length solution m; size = (M, 1)
    """
    # 1) Compute the general kernel Gg
    Wmi = np.linalg.inv(Wm)
    GWmiGt = np.matmul(np.matmul(G, Wmi), G.T)
    GWmiGt_condition_number = np.linalg.cond(GWmiGt)
    print(f"The condition number of matrix (G Wm^(-1) G^T) is {GWmiGt_condition_number}. \n"
          f"(Note: When the condition number is large (>e12), the inverse of this matrix is inaccurate (unstable).)")
    GWmiGt_inverse = np.linalg.inv(GWmiGt)
    Gg = np.matmul(np.matmul(Wmi, G.T), GWmiGt_inverse)
    # 2) Compute the weighted minimum length solution
    mest = model_prior + np.matmul(Gg, dobs - np.matmul(G, model_prior))
    print(f"Weighted minimum length solution is successfully computed.")
    return mest


##################################################################################################
###### Gm=d is mixed-determined, some models are over-determined, some are under-determined ######
##################################################################################################
def damped_least_square_solution(G, dobs, epsilon):
    """
    Compute the damped least-square solution for the linear inversion: mest = (G^T G + alpha^2 I)^(-1) G^T dobs
    When use this method, the problem should be mixed-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param epsilon: The damping factor (a value > 0, representing the relative importance of the solution length)
    :return: The damped least-square solution m; size = (M, 1)
    """
    # 1) Compute the general kernel Gg
    GtG = np.matmul(G.T, G)
    GtG_condition_number = np.linalg.cond(GtG)
    print(f"The condition number of matrix (G^T G) is {GtG_condition_number}")
    GtG_damped = GtG + epsilon**2 * np.identity(GtG.shape[0])
    GtG_damped_condition_number = np.linalg.cond(GtG_damped)
    print(f"The condition number of matrix (G^T G + ε^2 I) is {GtG_damped_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    GtG_damped_inverse = np.linalg.inv(GtG_damped)
    Gg = np.matmul(GtG_damped_inverse, G.T)
    # 2) Compute the damped least-square solution
    mest = np.matmul(Gg, dobs)
    print(f"Damped least square solution (ε={epsilon}) is successfully computed.")
    return mest


def weighted_damped_least_square_solution(G, dobs, model_prior, We, Wm, epsilon):
    """
    Compute the weighted damped least-square solution for the linear inversion:
    mest = (G^T We G + epsilon^2 Wm)^(-1) (G^T We dobs + epsilon^2 Wm m_prior)
    When use this method, the problem should be mixed-determined.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param We: The weight matrix (prior data covariance matrix) for the data vector d; size = (N, N)
    :param Wm: The weight matrix (prior model covariance matrix) for the model vector m; size = (M, M)
    :param epsilon: The damping factor (a value > 0, representing the relative importance of the solution length)
    :return: The weighted damped least-square solution m; size = (M, 1)
    """
    # 1) Compute the part 1: [G^T We G + epsilon^2 Wm]
    GtWe = np.matmul(G.T, We)
    GtWeG = np.matmul(GtWe, G)
    GtWed = np.matmul(GtWe, dobs)
    Gg1 = GtWeG + epsilon**2 * Wm
    Gg1_condition_number = np.linalg.cond(Gg1)
    print(f"The condition number of matrix (G^T We G + ε^2 Wm) is {Gg1_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    Gg1_inverse = np.linalg.inv(Gg1)
    # 2) Compute the part 2: [G^T We G dobs + epsilon^2 Wm m_prior]
    Wmm_prior = np.matmul(Wm, model_prior)
    Gg2 = GtWed + epsilon**2 * Wmm_prior
    # 3) Compute the weighted damped least-square solution
    mest = np.matmul(Gg1_inverse, Gg2)
    print(f"Weighted damped least square solution (ε={epsilon}) is successfully computed.")
    return mest


##################################################################################################
#######  Gm=d is mixed-determined, the problem solved in the Bayesian inference perspective ######
##################################################################################################
def bayesian_least_square_solution_formula1(G, dobs, model_prior, Cd, Cm):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance): Formula (6.8) and (6.9) in Andreas Fichtner's Book, 2021.
    Note: posterior mean model = maximum likelihood model -> We are in the Bayesian inference perspective.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    Cd_inverse = np.linalg.inv(Cd)
    Cm_inverse = np.linalg.inv(Cm)
    # 1) Compute the 1st term in (6.8):
    Gt_Cdi_G = np.matmul(np.matmul(G.T, Cd_inverse), G)
    Gt_Cdi_G_Cmi = Gt_Cdi_G + Cm_inverse
    Gt_Cdi_G_Cmi_condition_number = np.linalg.cond(Gt_Cdi_G_Cmi)
    print(f"The condition number of matrix (G^T Cd^(-1) G + Cm^(-1)) is {Gt_Cdi_G_Cmi_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    Gt_Cdi_G_Cmi_inverse = np.linalg.inv(Gt_Cdi_G_Cmi)  # The same as 'posterior model covariance matrix'
    # 2) Compute the 2nd term in (6.8):
    Gt_Cdi_dobs = np.matmul(np.matmul(G.T, Cd_inverse), dobs)
    Cmi_model_prior = np.matmul(Cm_inverse, model_prior)
    Gt_Cdi_dobs_Cmi_model_prior = Gt_Cdi_dobs + Cmi_model_prior
    # 3) Final result:
    posterior_mean_model = np.matmul(Gt_Cdi_G_Cmi_inverse, Gt_Cdi_dobs_Cmi_model_prior)
    posterior_model_covariance = Gt_Cdi_G_Cmi_inverse
    print(f"Bayesian least square solution is successfully computed.")
    return posterior_mean_model, posterior_model_covariance


def bayesian_least_square_solution_formula2(G, dobs, model_prior, Cd, Cm):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance): Formula (6.13) and (6.14) in Andreas Fichtner's Book, 2021.
    Note: posterior mean model = maximum likelihood model -> We are in the Bayesian inference perspective.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    # 1) Prepare some terms
    Cm_Gt = np.matmul(Cm, G.T)
    dobs_G_model_prior = dobs - np.matmul(G, model_prior)  # the last term for posterior mean
    G_Cm = np.matmul(G, Cm)                                # the last term for posterior covarianceS
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    Cd_G_Cm_Gt = Cd + G_Cm_Gt
    # Check the condition number of matrix (Cd + G Cm G^T)
    Cd_G_Cm_Gt_condition_number = np.linalg.cond(Cd_G_Cm_Gt)
    print(f"The condition number of matrix (Cd + G Cm G^T) is {Cd_G_Cm_Gt_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    # 2) Don't compute the inverse of Cd_G_Cm_Gt directly, we solve the linear system of equations
    last_2_term_mean = np.linalg.solve(Cd_G_Cm_Gt, dobs_G_model_prior)
    last_2_term_covariance = np.linalg.solve(Cd_G_Cm_Gt, G_Cm)
    # 3) Compute the two update terms for model and covariance
    update_term_mean = np.matmul(Cm_Gt, last_2_term_mean)
    update_term_covariance = np.matmul(Cm_Gt, last_2_term_covariance)
    # 4) Compute the posterior mean model and covariance matrix
    posterior_mean_model = model_prior + update_term_mean
    posterior_model_covariance = Cm - update_term_covariance
    print(f"Bayesian least square solution is successfully computed.")
    return posterior_mean_model, posterior_model_covariance


def bayesian_least_square_solution_uniform_model_prior(G, dobs, Cd):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance) with uniform model prior: Formula (6.26)
    Note 1: posterior mean model = maximum likelihood model -> We are in the Bayesian inference perspective.
    Note 2: This is an infinite range uniform distribution model prior.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    """
    # 1) Prepare some terms
    Gt_Cdi_G = np.matmul(np.matmul(G.T, np.linalg.inv(Cd)), G)
    Gt_Cdi_dobs = np.matmul(np.matmul(G.T, np.linalg.inv(Cd)), dobs)
    # 2) Check the condition number of matrix (G^T Cd^(-1) G)
    Gt_Cdi_G_condition_number = np.linalg.cond(Gt_Cdi_G)
    print(f"The condition number of matrix (G^T Cd^(-1) G) is {Gt_Cdi_G_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    # 3) Compute the posterior mean model and covariance matrix
    posterior_mean_model = np.linalg.solve(Gt_Cdi_G, Gt_Cdi_dobs)
    posterior_model_covariance = np.linalg.inv(Gt_Cdi_G)
    print(f"Bayesian least square solution with uniform model prior is successfully computed.")
    return posterior_mean_model, posterior_model_covariance


##################################################################################################
#######  Gm=d is mixed-determined, the problem solved in the Bayesian inference perspective ######
#######  When elements in G are tiny, we should introduce regularization term to stabilize  ######
######   1) We add the regularization term to the data covariance matrix: Cd + lambda * I   ######
##################################################################################################
def bayesian_least_square_solution_tikhonov_regularization_cd(G, dobs, model_prior, Cd, Cm, lambda_reg):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance) with Tikhonov regularization. We add the regularization term to the data covariance matrix: Cd + lambda * I
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    # 1) Compute G * Cm * G^T and the regularization term lambda * I
    G_Cm_Gt = np.matmul(np.matmul(G, Cm), G.T)
    reg_term = lambda_reg * np.eye(G_Cm_Gt.shape[0])
    # Compute the modified matrix Cd + G * Cm * G^T + lambda * I
    Cd_G_Cm_Gt_reg = Cd + G_Cm_Gt + reg_term
    Cd_G_Cm_Gt_reg_condition_number = np.linalg.cond(Cd_G_Cm_Gt_reg)
    print(f"The condition number of matrix (Cd + G Cm G^T + lambda I) is {Cd_G_Cm_Gt_reg_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    Cd_G_Cm_Gt_reg_inverse = np.linalg.inv(Cd_G_Cm_Gt_reg)
    # 2) Compute the new general kernel Gg
    Gg = np.matmul(np.matmul(Cm, G.T), Cd_G_Cm_Gt_reg_inverse)
    # 3) Compute the new posterior mean model
    posterior_mean_model = model_prior + np.matmul(Gg, dobs - np.matmul(G, model_prior))
    # 4) Compute the new posterior model covariance matrix
    posterior_model_covariance = Cm - np.matmul(np.matmul(Gg, G), Cm)
    print(f"Bayesian least square solution with Tikhonov regularization added to prior data covariance is successfully computed.")
    return posterior_mean_model, posterior_model_covariance


##################################################################################################
#######  Gm=d is mixed-determined, the problem solved in the Bayesian inference perspective ######
#######  When elements in G are tiny, we should introduce regularization term to stabilize  ######
######   2) We add the regularization term to the model covariance matrix: Cm + lambda * I  ######
##################################################################################################
def bayesian_least_square_solution_tikhonov_regularization_cm(G, dobs, model_prior, Cd, Cm, lambda_reg):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance) with Tikhonov regularization. We add the regularization term to the model covariance matrix: Cm + lambda * I
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param lambda_reg: The regularization parameter lambda
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    # 1) Add the regularization term lambda * I to Cm
    Cm_reg = Cm + lambda_reg * np.eye(Cm.shape[0])
    # 2) Compute G * Cm_reg * G^T
    G_Cm_reg_Gt = np.matmul(np.matmul(G, Cm_reg), G.T)
    # 3) Compute the modified matrix Cd + G * Cm_reg * G^T
    Cd_G_Cm_reg_Gt = Cd + G_Cm_reg_Gt
    Cd_G_Cm_reg_Gt_condition_number = np.linalg.cond(Cd_G_Cm_reg_Gt)
    print(f"The condition number of matrix (Cd + G Cm_reg G^T) is {Cd_G_Cm_reg_Gt_condition_number} \n"
          f"(Note: If the condition number is large (>e12), the inverse of this matrix is not accurate (unstable).)")
    Cd_G_Cm_reg_Gt_inverse = np.linalg.inv(Cd_G_Cm_reg_Gt)
    # 4) Compute the general kernel Gg
    Gg = np.matmul(np.matmul(Cm_reg, G.T), Cd_G_Cm_reg_Gt_inverse)
    # 5) Compute the posterior mean model
    posterior_mean_model = model_prior + np.matmul(Gg, dobs - np.matmul(G, model_prior))
    # 6) Compute the posterior model covariance matrix
    posterior_model_covariance = Cm_reg - np.matmul(np.matmul(Gg, G), Cm_reg)
    print(f"Bayesian least square solution with Tikhonov regularization added to prior model covariance is successfully computed.")
    return posterior_mean_model, posterior_model_covariance


##################################################################################################
#######  Gm=d is mixed-determined, the problem solved in the Bayesian inference perspective     ##
#######  Now we iteratively improve the solution: use posterior model / covariance as new prior ##
##################################################################################################
def iterative_bayesian_least_square_solution(G, dobs, initial_model_prior, Cd, initial_Cm, max_iterations=10):
    """
    Compute the Bayesian least-square solution (posterior mean and model covariance) iteratively from a model prior.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param initial_model_prior: The initial prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param initial_Cm: The initial prior model covariance matrix; size = (M, M)
    :param max_iterations: The maximum number of iterations
    :return: The final posterior mean model m, size = (M, 1); The final posterior model covariance matrix Cm, size = (M, M)
    """
    current_model_prior = initial_model_prior
    current_Cm = initial_Cm

    for iteration in range(max_iterations):
        # Compute Bayesian least-square solution: call the original function
        posterior_mean_model, posterior_model_covariance = bayesian_least_square_solution_formula2(
            G=G, dobs=dobs, model_prior=current_model_prior, Cd=Cd, Cm=current_Cm
        )

        # Update the prior model and covariance matrix for the next iteration
        current_model_prior = posterior_mean_model
        current_Cm = posterior_model_covariance
        print(f"Iteration {iteration + 1}: Posterior model mean and covariance matrix updated.")

    return current_model_prior, current_Cm


##################################################################################################
########################  Gm=d: Non-negative least-square solution (nnls) ########################
########################  Cholesky decomposition of Cm and Cd -> A        ########################
##################################################################################################
def bayesian_non_negative_least_square_solution(G, dobs, model_prior, Cd, Cm):
    """
    Compute the Bayesian non-negative least-square solution with a non-negative constraint.
    Note: actually we use truncated Gaussian distribution to approximate the non-negative constraint.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    # 1) Compute the Cholesky decomposition of Cd and Cm
    Ld = np.linalg.cholesky(Cd)
    Lm = np.linalg.cholesky(Cm)
    # 2) Compute the matrix A
    Ld_inverse = np.linalg.inv(Ld)
    Lm_inverse = np.linalg.inv(Lm)
    Ld_inverse_G = np.matmul(Ld_inverse, G)                        # size = (N, M)
    A = np.concatenate((Ld_inverse_G, Lm_inverse), axis=0)  # size = (N+M, M)
    # 3) Compute the vector b
    Ld_inverse_dobs = np.matmul(Ld_inverse, dobs)
    Lm_inverse_model_prior = np.matmul(Lm_inverse, model_prior)
    b = np.concatenate((Ld_inverse_dobs, Lm_inverse_model_prior), axis=0)  # size = (N+M, 1)
    # 4) Compute the tolerance
    norm_A_1 = np.linalg.norm(A, ord=1)
    atol_value = max(A.shape) * norm_A_1 * np.spacing(1.)
    print("The tolerance value for nnls is (by default): ", atol_value)
    # 5) Compute the non-negative least-square solution
    mest, residual = nnls(A=A, b=b, maxiter=10000, atol=atol_value)
    posterior_mean_model = mest
    # 6) Compute the posterior model covariance matrix
    Cd_inverse = np.linalg.inv(Cd)
    Cm_inverse = np.linalg.inv(Cm)
    Gt_Cdi_G = np.matmul(np.matmul(G.T, Cd_inverse), G)
    Gt_Cdi_G_Cmi = Gt_Cdi_G + Cm_inverse
    posterior_mean_cov = np.linalg.inv(Gt_Cdi_G_Cmi)
    # 7) Return
    # Note: here we still use the same names 'posterior_mean_model' and 'posterior_mean_cov' for the BNNLSM solution,
    #       but note that they don't have the same meaning as the Bayesian least-square solution !!!
    return posterior_mean_model, posterior_mean_cov


##################################################################################################
########################  Gm=d: Bounded least-square solution (bls)       ########################
########################  Cholesky decomposition of Cm and Cd -> A        ########################
##################################################################################################
def bayesian_bounded_least_square_solution(G, dobs, model_prior, Cd, Cm, bounds):
    """
    Compute the Bayesian bounded least-square solution with given bounds.
    Note: actually we use truncated Gaussian distribution to approximate the non-negative constraint.
    :param G: The kernel matrix G; size = (N, M)
    :param dobs: The observed data vector d; size = (N, 1)
    :param model_prior: The prior model vector m_prior; size = (M, 1)
    :param Cd: The prior data covariance matrix; size = (N, N)
    :param Cm: The prior model covariance matrix; size = (M, M)
    :param bounds: Bounds on the solution vector x; a tuple (min, max)
    :return: The posterior mean model m, size = (M, 1); The posterior model covariance matrix Cm, size = (M, M)
    """
    # 1) Compute the Cholesky decomposition of Cd and Cm
    Ld = np.linalg.cholesky(Cd)
    Lm = np.linalg.cholesky(Cm)
    # 2) Compute the matrix A
    Ld_inverse = np.linalg.inv(Ld)
    Lm_inverse = np.linalg.inv(Lm)
    Ld_inverse_G = np.matmul(Ld_inverse, G)                        # size = (N, M)
    A = np.concatenate((Ld_inverse_G, Lm_inverse), axis=0)  # size = (N+M, M)
    # 3) Compute the vector b
    Ld_inverse_dobs = np.matmul(Ld_inverse, dobs)
    Lm_inverse_model_prior = np.matmul(Lm_inverse, model_prior)
    b = np.concatenate((Ld_inverse_dobs, Lm_inverse_model_prior), axis=0)  # size = (N+M, 1)
    # 4) Compute the bounded least-square solution
    result = lsq_linear(A=A, b=b, bounds=bounds, max_iter=10000)
    posterior_mean_model = result.x
    # 5) Compute the posterior model covariance matrix
    Cd_inverse = np.linalg.inv(Cd)
    Cm_inverse = np.linalg.inv(Cm)
    Gt_Cdi_G = np.matmul(np.matmul(G.T, Cd_inverse), G)
    Gt_Cdi_G_Cmi = Gt_Cdi_G + Cm_inverse
    posterior_mean_cov = np.linalg.inv(Gt_Cdi_G_Cmi)
    # 6) Return
    # Note: here we still use the same names 'posterior_mean_model' and 'posterior_mean_cov' for the BNNLSM solution,
    #       but note that they don't have the same meaning as the Bayesian least-square solution !!!
    return posterior_mean_model, posterior_mean_cov





