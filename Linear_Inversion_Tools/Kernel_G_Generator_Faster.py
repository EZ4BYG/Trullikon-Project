from math import ceil
import numpy as np
import pickle
import copy
from Forward_Tools.GeertsmaSol_JP import GeertsmaSol_JP_py
from multiprocessing import Pool


def compute_distance_matrix(r_max, dx, dy, R, num_subs, shift_x=0.0, shift_y=0.0):
    """
    Compute the distance matrix between the sub-reservoirs centers and observation points. This matrix has the same shape as the kernel matrix G.
    :param r_max: The radius of the observation on the surface
    :param dx: The interval of the observation points in the x direction
    :param dy: The interval of the observation points in the y direction
    :param R: The radius of the original reservoir
    :param num_subs: The number of sub-reservoirs equally divided in the x or y direction
    :param shift_x: The shift in the x direction for the slight change in the observation points
    :param shift_y: The shift in the y direction for the slight change in the observation points
    :return: The distance matrix; size = (N, M), where N is the number of observation points, and M is the number of sub-reservoirs.
    """
    # 1) Observation points' coordinates
    obs_x = np.arange(-r_max, r_max + dx, dx) + shift_x
    obs_y = np.arange(-r_max, r_max + dy, dy) + shift_y
    Obs_X, Obs_Y = np.meshgrid(obs_x, obs_y)

    # 2) Sub-reservoirs' center coordinates
    sub_R = R / num_subs
    offset_max = R - sub_R
    center_x = np.linspace(-offset_max, offset_max, num_subs)
    center_y = np.linspace(-offset_max, offset_max, num_subs)
    Center_X, Center_Y = np.meshgrid(center_x, center_y)

    # 3) Flatten them using a same order
    Obs_X_Flatten = Obs_X.flatten()
    Obs_Y_Flatten = Obs_Y.flatten()
    Center_X_Flatten = Center_X.flatten()
    Center_Y_Flatten = Center_Y.flatten()

    # 4) Compute the distance matrix
    num_points = Obs_X.size
    num_centers = Center_X.size
    dist_matrix = np.zeros((num_points, num_centers))
    # Note: loop each column
    for j in range(num_centers):
        dist_matrix[:, j] = np.sqrt((Obs_X_Flatten - Center_X_Flatten[j]) ** 2 + (Obs_Y_Flatten - Center_Y_Flatten[j]) ** 2)

    # 5) Return the distance matrix
    return dist_matrix


def tilt_convert_xy(uz, uz_dx, uz_dy):
    """
    Convert the tilt components in (x,y) coordinate
    :param uz: The vertical displacement at all observation points. The unit is [m].
    :param uz_dx: The vertical displacement at all observation points with a slight shift in the x direction. The unit is [m].
    :param uz_dy: The vertical displacement at all observation points with a slight shift in the y direction. The unit is [m].
    :return: The tilt components in (x,y) coordinate. The unit of the tilts is [micro-radian].
    """
    # 1) Tilt_x
    duz_dx = (uz_dx - uz) / 1e-3
    tilt_x = np.arctan(-duz_dx) * 1e6  # Convert to micro-radian
    # 2) Tilt_y
    duz_dy = (uz_dy - uz) / 1e-3
    tilt_y = np.arctan(-duz_dy) * 1e6  # Convert to micro-radian
    # 3) Return the tilt components in (x,y) coordinate
    return tilt_x, tilt_y


def compute_observations(args):
    """
    """
    # 1) Extract the necessary parameters
    model_for_G, r, r_dx, r_dy, sub_R = args
    model = copy.deepcopy(model_for_G)

    # 2.1) Compute the original displacement
    uz, ur = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                               alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                               p=model['P'], r=r, R=sub_R,
                               iL_pp=model['iL_pp'], RHO=model['RHO'],
                               VS=model['VS'], VP=model['VP'])
    # 2.2) Prepare for tiltx
    uz_dx, ur_dx = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                                     alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                                     p=model['P'], r=r_dx, R=sub_R,
                                     iL_pp=model['iL_pp'], RHO=model['RHO'],
                                     VS=model['VS'], VP=model['VP'])
    # 2.3) Prepare for tilty
    uz_dy, ur_dy = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                                     alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                                     p=model['P'], r=r_dy, R=sub_R,
                                     iL_pp=model['iL_pp'], RHO=model['RHO'],
                                     VS=model['VS'], VP=model['VP'])
    # 2.4) Final processing
    uz = np.real(uz)[0].reshape(r.shape)
    ur = np.real(ur)[0].reshape(r.shape)
    uz_dx = np.real(uz_dx)[0].reshape(r.shape)
    uz_dy = np.real(uz_dy)[0].reshape(r.shape)

    # 3) Compute the tiltx and tilty
    tilt_x, tilt_y = tilt_convert_xy(uz, uz_dx, uz_dy)

    # 4) Return the results
    return uz, ur, tilt_x, tilt_y


def kernel_g_generator_faster(model_for_G, auxiliary_parameters, save=True):
    """
    Generate all Kernel G (G_uz, G_ur, G_tiltx, G_tilty) numerically for the linear inversion: d=Gm
    """
    # 1) Extract the necessary parameters
    r_max = auxiliary_parameters['r_max']
    dx = auxiliary_parameters['dx']
    dy = auxiliary_parameters['dy']
    num_subs = auxiliary_parameters['num_subs']
    process_num = auxiliary_parameters['process_num']
    R = model_for_G['R']
    # Compute the radius of each sub-reservoir
    sub_R = R / num_subs

    # 2) Compute the distance matrix
    dist_matrix = compute_distance_matrix(r_max, dx, dy, R, num_subs)
    num_points = dist_matrix.shape[0]
    num_centers = dist_matrix.shape[1]
    print(f'The size of Kernel G is {num_points}x{num_centers}. The Kernel G generation begins:')
    # These two matrices are used to compute the partial derivatives (for tiltx and tilty)
    dist_matrix_dx = compute_distance_matrix(r_max, dx, dy, R, num_subs, shift_x=1e-3)
    dist_matrix_dy = compute_distance_matrix(r_max, dx, dy, R, num_subs, shift_y=1e-3)

    # 3) Loop each column to compute the corresponding column of the kernel matrix G
    tasks = [(model_for_G, dist_matrix[:, j], dist_matrix_dx[:, j], dist_matrix_dy[:, j], sub_R) for j in range(num_centers)]
    with Pool(processes=process_num) as p:
        results = p.map(compute_observations, tasks)

    # 4) Generate different kernel matrix G based on the results
    G_uz = np.zeros((num_points, num_centers))
    G_ur = np.zeros((num_points, num_centers))
    G_tiltx = np.zeros((num_points, num_centers))
    G_tilty = np.zeros((num_points, num_centers))
    for j, (uz, ur, tilt_x, tilt_y) in enumerate(results):
        G_uz[:, j] = uz / model_for_G['P_reservoir'] * 1e3   # Unit: [mm]
        G_ur[:, j] = ur / model_for_G['P_reservoir'] * 1e3   # Unit: [mm]
        G_tiltx[:, j] = tilt_x / model_for_G['P_reservoir']  # Unit: [micro-radian]
        G_tilty[:, j] = tilt_y / model_for_G['P_reservoir']  # Unit: [micro-radian]

    # 5) Save matrices if needed
    if save:
        np.save(f"G_uz_{num_points}x{num_centers}.npy", G_uz)
        np.save(f"G_ur_{num_points}x{num_centers}.npy", G_ur)
        np.save(f"G_tiltx_{num_points}x{num_centers}.npy", G_tiltx)
        np.save(f"G_tilty_{num_points}x{num_centers}.npy", G_tilty)

    # 6) Return all G
    print("All Kernel G (G_uz, G_ur, G_tiltx, G_tilty) generation are successfully completed.")
    return G_uz, G_ur, G_tiltx, G_tilty

