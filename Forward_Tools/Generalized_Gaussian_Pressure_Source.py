import numpy as np
import matplotlib.pyplot as plt


def generalized_gaussian_distribution(x, y, x0, y0, alpha_x, alpha_y, beta_x, beta_y, amplitude):
    """
    Auxiliary Function: Compute the 2D Generalized Gaussian distribution with different shape parameters in x and y directions.
    :param x: x-index
    :param y: y-index
    :param x0: x-center of the distribution index
    :param y0: y-center of the distribution index
    :param alpha_x: scale parameter in x-direction
    :param alpha_y: scale parameter in y-direction
    :param beta_x: shape parameter controlling the shape of the distribution in x-direction
    :param beta_y: shape parameter controlling the shape of the distribution in y-direction
    :param amplitude: amplitude of the distribution at the center
    """
    return amplitude * np.exp(-((np.abs(x - x0) / alpha_x) ** beta_x + (np.abs(y - y0) / alpha_y) ** beta_y))


def assign_generalized_gaussian_pressure(num_subs, centers: list[tuple], alphas: list[tuple], betas: list[tuple], amplitudes: list[float]):
    """
    Use the Generalized Gaussian distribution to assign pressure sources to the sub reservoirs.
    :param num_subs: number of sub reservoirs in the x or y directions
    :param centers: list of tuples containing the center of the distribution(s)
    :param alphas: list of tuples containing the scale parameters in x and y directions of the distribution(s)
    :param betas: list of tuples containing the shape parameters in x and y directions of the distribution(s)
    :param amplitudes: list of amplitudes of the distribution at the center
    :return: list of each sub reservoir's pressure
    """
    # Initialize the pressure sources
    pressure_sources = np.zeros((num_subs, num_subs))
    x = np.arange(num_subs)
    y = np.arange(num_subs)
    X, Y = np.meshgrid(x, y)

    # Assign the Generalized Gaussian pressure sources to the sub reservoirs and sum them up
    for (center_x, center_y), (alpha_x, alpha_y), (beta_x, beta_y), amplitude in zip(centers, alphas, betas, amplitudes):
        pressure_sources += generalized_gaussian_distribution(X, Y, center_x, center_y, alpha_x, alpha_y, beta_x, beta_y, amplitude)

    # Convert the 2D array to list
    pressure_list = pressure_sources.flatten().tolist()
    return pressure_list


# Test and visualization in the main function
if __name__ == '__main__':
    # 1) Global parameters: An example with 2 small regions
    # Note: beta = 2 is the Gaussian distribution,
    #       beta < 2 is sub-Gaussian distribution (slower decay), beta > 2 is super-Gaussian distribution (faster decay)
    num_subs = 125                   # 125 sub-reservoirs in the x or y directions. Total number is 125**2
    centers = [(20, 23), (75, 75)]   # 2 small regions
    alphas = [(9, 8), (15, 18)]      # [(alpha_x1, alpha_y1), (alpha_x2, alpha_y2)]
    betas = [(1.5, 2.0), (1.8, 1.2)] # [(beta_x1, beta_y1), (beta_x2, beta_y2)]
    amplitudes = [10, 15]            # [amplitude1, amplitude2]

    # 2) Call the function
    pressure_list = assign_generalized_gaussian_pressure(num_subs, centers, alphas, betas, amplitudes)

    # 3) Visualization
    pressure_matrix = np.array(pressure_list).reshape((num_subs, num_subs))
    plt.imshow(pressure_matrix, extent=[0, num_subs, 0, num_subs], origin='lower', cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.scatter(*zip(*centers), color='red', marker='x')  # Mark the center of the distribution
    plt.title('Pressure Distribution (Index Units)')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.tight_layout()
    plt.show()
