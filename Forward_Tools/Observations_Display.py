import matplotlib.pyplot as plt
import numpy as np


def plot_observations(X, Y, uz_total, ur_total, tilt_x_total, tilt_y_total,
                      num_levels=100, file_name='uz_ur_tiltx_tilty.png'):
    """
    Plot 4 observations: uz, ur, tilt_x, tilt_y on 4 subplots.
    :param X: The meshgrid in x direction
    :param Y: The meshgrid in y direction
    :param uz_total: The vertical displacement at all observation points
    :param ur_total: The horizontal displacement at all observation points
    :param tilt_x_total: The horizontal tilt at all observation points
    :param tilt_y_total: The vertical tilt at all observation points
    :param num_levels: The number of levels in the contour plot
    :param file_name: The file name to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Vertical Displacement uz
    im1 = axes[0, 0].contourf(X, Y, uz_total*1e3, levels=num_levels, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Displacement [mm]', fontsize=12)
    axes[0, 0].set_title('Vertical Displacement (uz) on the Surface', fontsize=14)
    axes[0, 0].set_xlabel('X [m]', fontsize=12)
    axes[0, 0].set_ylabel('Y [m]', fontsize=12)

    # Horizontal Displacement ur
    im2 = axes[0, 1].contourf(X, Y, ur_total*1e3, levels=num_levels, cmap='viridis')
    cbar2 = fig.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('Displacement [mm]', fontsize=12)
    axes[0, 1].set_title('Horizontal Displacement (ur) on the Surface', fontsize=14)
    axes[0, 1].set_xlabel('X [m]', fontsize=12)
    axes[0, 1].set_ylabel('Y [m]', fontsize=12)

    # Tilt_x
    im3 = axes[1, 0].contourf(X, Y, tilt_x_total, levels=num_levels, cmap='viridis')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('Tilt [microradian]', fontsize=12)
    axes[1, 0].set_title('X-direction Tilt (tilt_x) on the Surface', fontsize=14)
    axes[1, 0].set_xlabel('X [m]', fontsize=12)
    axes[1, 0].set_ylabel('Y [m]', fontsize=12)

    # Tilt_y
    im4 = axes[1, 1].contourf(X, Y, tilt_y_total, levels=num_levels, cmap='viridis')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1])
    cbar4.set_label('Tilt [microradian]', fontsize=12)
    axes[1, 1].set_title('Y-direction Tilt (tilt_y) on the Surface', fontsize=14)
    axes[1, 1].set_xlabel('X [m]', fontsize=12)
    axes[1, 1].set_ylabel('Y [m]', fontsize=12)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_1st_derivative_observations(X, Y, grad_uz_total, grad_ur_total, grad_tilt_x_total, grad_tilt_y_total,
                                     num_levels=100, file_name='uz_ur_tiltx_tilty_1st_gradients.png'):
    """
    Plot 4 observations' 1st derivatives: grad_uz, grad_ur, grad_tilt_x, grad_tilt_y on 4 subplots.
    :param X: The meshgrid in x direction
    :param Y: The meshgrid in y direction
    :param grad_uz_total: Gradient of uz in both x and y directions
    :param grad_ur_total: Gradient of ur in both x and y directions
    :param grad_tilt_x_total: Gradient of tilt_x in both x and y directions
    :param grad_tilt_y_total: Gradient of tilt_y in both x and y directions
    :param num_levels: The number of levels in the contour plot
    :param file_name: The file name to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Gradient of uz
    im1 = axes[0, 0].contourf(X, Y, np.sqrt(grad_uz_total[0]**2 + grad_uz_total[1]**2), levels=num_levels, cmap='RdBu')
    cbar1 = fig.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Gradient [mm/m]', fontsize=12)
    axes[0, 0].set_title('1st derivative magnitude of Vertical Displacement (uz)', fontsize=14)
    axes[0, 0].set_xlabel('X [m]', fontsize=12)
    axes[0, 0].set_ylabel('Y [m]', fontsize=12)

    # Gradient of ur
    im2 = axes[0, 1].contourf(X, Y, np.sqrt(grad_ur_total[0]**2 + grad_ur_total[1]**2), levels=num_levels, cmap='RdBu')
    cbar2 = fig.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('Gradient [mm/m]', fontsize=12)
    axes[0, 1].set_title('1st derivative magnitude of Horizontal Displacement (ur)', fontsize=14)
    axes[0, 1].set_xlabel('X [m]', fontsize=12)
    axes[0, 1].set_ylabel('Y [m]', fontsize=12)

    # Gradient of tilt_x
    im3 = axes[1, 0].contourf(X, Y, np.sqrt(grad_tilt_x_total[0]**2 + grad_tilt_x_total[1]**2), levels=num_levels, cmap='RdBu')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('Gradient [microradian/m]', fontsize=12)
    axes[1, 0].set_title('1st derivative magnitude of X-direction Tilt (tilt_x)', fontsize=14)
    axes[1, 0].set_xlabel('X [m]', fontsize=12)
    axes[1, 0].set_ylabel('Y [m]', fontsize=12)

    # Gradient of tilt_y
    im4 = axes[1, 1].contourf(X, Y, np.sqrt(grad_tilt_y_total[0]**2 + grad_tilt_y_total[1]**2), levels=num_levels, cmap='RdBu')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1])
    cbar4.set_label('Gradient [microradian/m]', fontsize=12)
    axes[1, 1].set_title('1st derivative magnitude of Y-direction Tilt (tilt_y)', fontsize=14)
    axes[1, 1].set_xlabel('X [m]', fontsize=12)
    axes[1, 1].set_ylabel('Y [m]', fontsize=12)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_2nd_derivative_observations(X, Y, second_grad_uz_total, second_grad_ur_total, second_grad_tilt_x_total, second_grad_tilt_y_total,
                                     num_levels=100, file_name='uz_ur_tiltx_tilty_2nd_gradients.png'):
    """
    Plot 4 observations' 2nd derivatives: second_grad_uz, second_grad_ur, second_grad_tilt_x, second_grad_tilt_y on 4 subplots.
    :param X: The meshgrid in x direction
    :param Y: The meshgrid in y direction
    :param second_grad_uz_total: 2nd gradient of uz in both x and y directions
    :param second_grad_ur_total: 2nd gradient of ur in both x and y directions
    :param second_grad_tilt_x_total: 2nd gradient of tilt_x in both x and y directions
    :param second_grad_tilt_y_total: 2nd gradient of tilt_y in both x and y directions
    :param num_levels: The number of levels in the contour plot
    :param file_name: The file name to save the plot
    """
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))

    # 1) Combine 2nd gradients for uz
    combined_uz_x = np.sqrt(second_grad_uz_total[0][0] ** 2 + second_grad_uz_total[0][1] ** 2)
    combined_uz_y = np.sqrt(second_grad_uz_total[1][0] ** 2 + second_grad_uz_total[1][1] ** 2)
    combined_uz_total = np.sqrt(second_grad_uz_total[0][0] ** 2 + second_grad_uz_total[0][1] ** 2 +
                                second_grad_uz_total[1][0] ** 2 + second_grad_uz_total[1][1] ** 2)

    # 2nd Gradient of uz
    im1 = axes[0, 0].contourf(X, Y, combined_uz_x, levels=50, cmap='RdBu')
    cbar1 = fig.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[0, 0].set_title('2nd derivative of Vertical Displacement (uz) - x direction', fontsize=14)
    axes[0, 0].set_xlabel('X [mm]', fontsize=12)
    axes[0, 0].set_ylabel('Y [mm]', fontsize=12)

    im2 = axes[0, 1].contourf(X, Y, combined_uz_y, levels=50, cmap='RdBu')
    cbar2 = fig.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[0, 1].set_title('2nd derivative of Vertical Displacement (uz) - y direction', fontsize=14)
    axes[0, 1].set_xlabel('X [mm]', fontsize=12)
    axes[0, 1].set_ylabel('Y [mm]', fontsize=12)

    im3 = axes[0, 2].contourf(X, Y, combined_uz_total, levels=50, cmap='RdBu')
    cbar3 = fig.colorbar(im3, ax=axes[0, 2])
    cbar3.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[0, 2].set_title('2nd derivative of Vertical Displacement (uz) - total', fontsize=14)
    axes[0, 2].set_xlabel('X [mm]', fontsize=12)
    axes[0, 2].set_ylabel('Y [mm]', fontsize=12)

    # 2) Combine 2nd gradients for ur
    combined_ur_x = np.sqrt(second_grad_ur_total[0][0] ** 2 + second_grad_ur_total[0][1] ** 2)
    combined_ur_y = np.sqrt(second_grad_ur_total[1][0] ** 2 + second_grad_ur_total[1][1] ** 2)
    combined_ur_total = np.sqrt(second_grad_ur_total[0][0] ** 2 + second_grad_ur_total[0][1] ** 2 +
                                second_grad_ur_total[1][0] ** 2 + second_grad_ur_total[1][1] ** 2)

    # 2nd Gradient of ur
    im4 = axes[1, 0].contourf(X, Y, combined_ur_x, levels=50, cmap='RdBu')
    cbar4 = fig.colorbar(im4, ax=axes[1, 0])
    cbar4.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[1, 0].set_title('2nd derivative of Horizontal Displacement (ur) - x direction', fontsize=14)
    axes[1, 0].set_xlabel('X [mm]', fontsize=12)
    axes[1, 0].set_ylabel('Y [mm]', fontsize=12)

    im5 = axes[1, 1].contourf(X, Y, combined_ur_y, levels=50, cmap='RdBu')
    cbar5 = fig.colorbar(im5, ax=axes[1, 1])
    cbar5.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[1, 1].set_title('2nd derivative of Horizontal Displacement (ur) - y direction', fontsize=14)
    axes[1, 1].set_xlabel('X [mm]', fontsize=12)
    axes[1, 1].set_ylabel('Y [mm]', fontsize=12)

    im6 = axes[1, 2].contourf(X, Y, combined_ur_total, levels=50, cmap='RdBu')
    cbar6 = fig.colorbar(im6, ax=axes[1, 2])
    cbar6.set_label('2nd Gradient [mm/m^2]', fontsize=12)
    axes[1, 2].set_title('2nd derivative of Horizontal Displacement (ur) - total', fontsize=14)
    axes[1, 2].set_xlabel('X [mm]', fontsize=12)
    axes[1, 2].set_ylabel('Y [mm]', fontsize=12)

    # 3) Combine 2nd gradients for tilt_x
    combined_tilt_x_x = np.sqrt(second_grad_tilt_x_total[0][0] ** 2 + second_grad_tilt_x_total[0][1] ** 2)
    combined_tilt_x_y = np.sqrt(second_grad_tilt_x_total[1][0] ** 2 + second_grad_tilt_x_total[1][1] ** 2)
    combined_tilt_x_total = np.sqrt(second_grad_tilt_x_total[0][0] ** 2 + second_grad_tilt_x_total[0][1] ** 2 +
                                    second_grad_tilt_x_total[1][0] ** 2 + second_grad_tilt_x_total[1][1] ** 2)

    # 2nd Gradient of tilt_x
    im7 = axes[2, 0].contourf(X, Y, combined_tilt_x_x, levels=50, cmap='RdBu')
    cbar7 = fig.colorbar(im7, ax=axes[2, 0])
    cbar7.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[2, 0].set_title('2nd derivative of X-direction Tilt (tilt_x) - x direction', fontsize=14)
    axes[2, 0].set_xlabel('X [mm]', fontsize=12)
    axes[2, 0].set_ylabel('Y [mm]', fontsize=12)

    im8 = axes[2, 1].contourf(X, Y, combined_tilt_x_y, levels=50, cmap='RdBu')
    cbar8 = fig.colorbar(im8, ax=axes[2, 1])
    cbar8.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[2, 1].set_title('2nd derivative of X-direction Tilt (tilt_x) - y direction', fontsize=14)
    axes[2, 1].set_xlabel('X [mm]', fontsize=12)
    axes[2, 1].set_ylabel('Y [mm]', fontsize=12)

    im9 = axes[2, 2].contourf(X, Y, combined_tilt_x_total, levels=50, cmap='RdBu')
    cbar9 = fig.colorbar(im9, ax=axes[2, 2])
    cbar9.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[2, 2].set_title('2nd derivative of X-direction Tilt (tilt_x) - total', fontsize=14)
    axes[2, 2].set_xlabel('X [mm]', fontsize=12)
    axes[2, 2].set_ylabel('Y [mm]', fontsize=12)

    # 4) Combine 2nd gradients for tilt_y
    combined_tilt_y_x = np.sqrt(second_grad_tilt_y_total[0][0] ** 2 + second_grad_tilt_y_total[0][1] ** 2)
    combined_tilt_y_y = np.sqrt(second_grad_tilt_y_total[1][0] ** 2 + second_grad_tilt_y_total[1][1] ** 2)
    combined_tilt_y_total = np.sqrt(second_grad_tilt_y_total[0][0] ** 2 + second_grad_tilt_y_total[0][1] ** 2 +
                                    second_grad_tilt_y_total[1][0] ** 2 + second_grad_tilt_y_total[1][1] ** 2)

    # 2nd Gradient of tilt_y
    im10 = axes[3, 0].contourf(X, Y, combined_tilt_y_x, levels=50, cmap='RdBu')
    cbar10 = fig.colorbar(im10, ax=axes[3, 0])
    cbar10.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[3, 0].set_title('2nd derivative of Y-direction Tilt (tilt_y) - x direction', fontsize=14)
    axes[3, 0].set_xlabel('X [mm]', fontsize=12)
    axes[3, 0].set_ylabel('Y [mm]', fontsize=12)

    im11 = axes[3, 1].contourf(X, Y, combined_tilt_y_y, levels=50, cmap='RdBu')
    cbar11 = fig.colorbar(im11, ax=axes[3, 1])
    cbar11.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[3, 1].set_title('2nd derivative of Y-direction Tilt (tilt_y) - y direction', fontsize=14)
    axes[3, 1].set_xlabel('X [mm]', fontsize=12)
    axes[3, 1].set_ylabel('Y [mm]', fontsize=12)

    im12 = axes[3, 2].contourf(X, Y, combined_tilt_y_total, levels=50, cmap='RdBu')
    cbar12 = fig.colorbar(im12, ax=axes[3, 2])
    cbar12.set_label('2nd Gradient [microradian/m^2]', fontsize=12)
    axes[3, 2].set_title('2nd derivative of Y-direction Tilt (tilt_y) - total', fontsize=14)
    axes[3, 2].set_xlabel('X [mm]', fontsize=12)
    axes[3, 2].set_ylabel('Y [mm]', fontsize=12)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()