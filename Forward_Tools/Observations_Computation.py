import numpy as np
import copy
from Forward_Tools.GeertsmaSol_JP import GeertsmaSol_JP_py  # Absolute path
# Multiprocessing
from multiprocessing import Pool


def compute_1st_gradients(X, Y, observation_data):
    """
    Compute the gradients of some kind of observation in the x and y directions.
    :param X: The meshgrid in x direction
    :param Y: The meshgrid in y direction
    :param observation_data: one kind of observation data
    """
    gradient_x, gradient_y = np.gradient(observation_data, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])
    return gradient_x, gradient_y


def compute_2nd_gradients(X, Y, observation_data):
    """
    Compute the 2nd gradients of some kind of observation in the x and y directions.
    :param X: The meshgrid in x direction
    :param Y: The meshgrid in y direction
    :param observation_data: one kind of observation data
    :return: The combined 2nd gradients in x and y directions
    """
    grad_x, grad_y = np.gradient(observation_data, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])
    grad_xx, grad_xy = np.gradient(grad_x, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])
    grad_yx, grad_yy = np.gradient(grad_y, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])
    return (grad_xx, grad_xy), (grad_yx, grad_yy)


def tilt_convert_xy(uz, uz_dx, uz_dy, dx=1e-3, dy=1e-3):
    """
    Convert the tilt components in (x,y) coordinate
    :param uz: The vertical displacement at all observation points. The unit is [m].
    :param uz_dx: The vertical displacement at all observation points with a slight shift in the x direction. The unit is [m].
    :param uz_dy: The vertical displacement at all observation points with a slight shift in the y direction. The unit is [m].
    :param dx: The small shift value in the x direction. The unit is [m].
    :param dy: The small shift value in the y direction. The unit is [m].
    :return: The tilt components in (x,y) coordinate. The unit of the tilts is [micro-radian].
    """
    # 1) Tilt_x
    duz_dx = (uz_dx - uz) / dx
    tilt_x_total = np.arctan(-duz_dx) * 1e6  # Convert to micro-radian
    # 2) Tilt_y
    duz_dy = (uz_dy - uz) / dy
    tilt_y_total = np.arctan(-duz_dy) * 1e6  # Convert to micro-radian
    # 3) Return the tilt components in (x,y) coordinate
    return tilt_x_total, tilt_y_total


def compute_sub_reservoir(args):
    """
    An auxiliary function will be called by 'grid_1d_model' function.
    """
    i, center, sub_pressure, sub_R, model_general, X, Y, X_dx, Y_dx, X_dy, Y_dy, print_info = args
    model = copy.deepcopy(model_general)
    # update the sub-reservoir's pressure
    model['P'][model['iL_pp'] - 1] = sub_pressure

    # Distance between all observation points and the center of current sub-reservoir
    # original
    centerx = X - center[0]
    centery = Y - center[1]
    r = np.sqrt(centerx ** 2 + centery ** 2).ravel()
    # for tilt_x
    centerx_dx = X_dx - center[0]
    centery_dx = Y_dx - center[1]
    r_dx = np.sqrt(centerx_dx ** 2 + centery_dx ** 2).ravel()
    # for tilt_y
    centerx_dy = X_dy - center[0]
    centery_dy = Y_dy - center[1]
    r_dy = np.sqrt(centerx_dy ** 2 + centery_dy ** 2).ravel()

    # Displacement
    # original
    uz, ur = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                               alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                               p=model['P'], r=r, R=sub_R,
                               iL_pp=model['iL_pp'], RHO=model['RHO'],
                               VS=model['VS'], VP=model['VP'])
    # for tilt_x
    uz_dx, ur_dx = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                                     alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                                     p=model['P'], r=r_dx, R=sub_R,
                                     iL_pp=model['iL_pp'], RHO=model['RHO'],
                                     VS=model['VS'], VP=model['VP'])
    # for tilt_y
    uz_dy, ur_dy = GeertsmaSol_JP_py(NL=model['NL'], G=model['G'], nu=model['Nu'],
                                     alpha=model['Alpha'], z_top=model['Z_top'], z_bot=model['Z_bot'],
                                     p=model['P'], r=r_dy, R=sub_R,
                                     iL_pp=model['iL_pp'], RHO=model['RHO'],
                                     VS=model['VS'], VP=model['VP'])

    # Final processing and return
    uz = np.real(uz)[0].reshape(X.shape)
    ur = np.real(ur)[0].reshape(X.shape)
    uz_dx = np.real(uz_dx)[0].reshape(X.shape)
    uz_dy = np.real(uz_dy)[0].reshape(X.shape)
    if print_info:
        print(f"{i}th sub_r: {center} finished!")
    return uz, ur, uz_dx, uz_dy


def grid_1d_model(R, num_subs, r_max, dx, dy, model_general, sub_pressures=None, process_num=4, print_info=True):
    """
    We only grid the reservoir layer; '1l' means 1 layer
    :param R: The radius of the original reservoir
    :param num_subs: The number of sub-reservoirs equally divided in the x or y direction
    :param r_max: The radius of the observation on the surface
    :param dx: The interval of the observation points in the x direction
    :param dy: The interval of the observation points in the y direction
    :param model_general: The general model parameters defined in a dictionary
    :param sub_pressures: The list of the sub-reservoirs pressures if num_subs != 0
    :param process_num: The number of processes for multiprocessing
    :param print_info: Whether to print the information
    :return: The grids and data. The unit of the displacements is [m], and the unit of the tilts is [micro-radian].
    """
    # 1) Check if the sub-reservoir's number is odd.
    if num_subs != 0 and num_subs % 2 == 0:
        raise ValueError("It's best for the number of sub-reservoirs to be an odd number")

    # 2) Observation points on the surface in (x,y) coordinate. Total number is len(X) * len(Y)
    x = np.arange(-r_max, r_max + dx, dx)
    y = np.arange(-r_max, r_max + dy, dy)
    X, Y = np.meshgrid(x, y)
    X_dx, Y_dx = X + 1e-3, Y
    X_dy, Y_dy = X, Y + 1e-3

    # 3) Compute center coordinates of each sub-reservoir in (x,y) coordinate
    sub_R = R / num_subs
    offset_max = R - sub_R
    # Note there: row-major order!
    centers = [(x, y) for y in np.linspace(-offset_max, offset_max, num_subs)
               for x in np.linspace(-offset_max, offset_max, num_subs)]

    # 4) Collect the arguments for each sub-reservoir
    args = [(i, center, sub_pressures[i], sub_R, model_general, X, Y, X_dx, Y_dx, X_dy, Y_dy, print_info) for i, center in
            enumerate(centers)]

    # 5) Multiprocessing to compute the displacements and tilts
    with Pool(processes=process_num) as pool:
        results = pool.map(compute_sub_reservoir, args)

    # 6) Sum up the displacements and tilts from all sub-reservoirs contributions
    uz_total = np.zeros_like(X, dtype=np.float64)
    ur_total = np.zeros_like(X, dtype=np.float64)
    uz_dx_total = np.zeros_like(X, dtype=np.float64)
    uz_dy_total = np.zeros_like(X, dtype=np.float64)
    for uz, ur, uz_dx, uz_dy in results:
        uz_total += uz
        ur_total += ur
        uz_dx_total += uz_dx
        uz_dy_total += uz_dy

    # 7) Call 'tilt_convert_xy' function
    tilt_x_total, tilt_y_total = tilt_convert_xy(uz_total, uz_dx_total, uz_dy_total, dx=1e-3, dy=1e-3)

    # 8) Return the grids and data
    if print_info:
        print('Displacements (uz + ur) and Tilts (tilt_x + tilt_y) computation is done!')
    return X, Y, uz_total, ur_total, tilt_x_total, tilt_y_total

