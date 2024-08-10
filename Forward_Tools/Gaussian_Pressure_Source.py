import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def gaussian_distribution(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    """
    Auxiliary Function: Compute the 2D Gaussian distribution.
    :param x: x-index
    :param y: y-index
    :param x0: x-center of the Gaussian distribution index
    :param y0: y-center of the Gaussian distribution index
    :param sigma_x: standard deviation in x-direction
    :param sigma_y: standard deviation in y-direction
    :param amplitude: amplitude of the Gaussian distribution at the center
    """
    return amplitude * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


def assign_gaussian_pressure(num_subs, centers: list[tuple], sigmas: list[tuple], amplitudes: list[float]):
    """
    Use the Gaussian distribution to assign pressure sources to the sub reservoirs.
    :param num_subs: number of sub reservoirs in the x or y directions
    :param centers: list of tuples containing the center of the Gaussian distribution(s)
    :param sigmas: list of tuples containing the standard deviation in x and y directions of the Gaussian distribution(s)
    :param amplitudes: list of amplitudes of the Gaussian distribution at the center
    :return: list of each sub reservoir's pressure
    """
    # Initialize the pressure sources
    pressure_sources = np.zeros((num_subs, num_subs))
    x = np.arange(num_subs)
    y = np.arange(num_subs)
    X, Y = np.meshgrid(x, y)

    # Assign the Gaussian pressure sources to the sub reservoirs and sum them up
    for (center_x, center_y), (sigma_x, sigma_y), amplitude in zip(centers, sigmas, amplitudes):
        pressure_sources += gaussian_distribution(X, Y, center_x, center_y, sigma_x, sigma_y, amplitude)

    # Convert the 2D array to list
    pressure_list = pressure_sources.flatten().tolist()
    return pressure_list


def assign_gaussian_pressure_with_zero_zone(num_subs, centers: list[tuple], sigmas: list[tuple], amplitudes: list[float], zero_zone_vertices: list[tuple]):
    """
    Use the Gaussian distribution to assign pressure sources to the sub reservoirs, and assign zero pressure to the specified zone.
    :param num_subs: number of sub reservoirs in the x or y directions
    :param centers: list of tuples containing the center of the Gaussian distribution(s)
    :param sigmas: list of tuples containing the standard deviation in x and y directions of the Gaussian distribution(s)
    :param amplitudes: list of amplitudes of the Gaussian distribution at the center
    :param zero_zone_vertices: list of three tuples representing the vertices of the triangle that should be zeroed out
    :return: list of each sub reservoir's pressure
    """
    # Initialize the pressure sources
    pressure_sources = np.zeros((num_subs, num_subs))
    x = np.arange(num_subs)
    y = np.arange(num_subs)
    X, Y = np.meshgrid(x, y)

    # Assign the Gaussian pressure sources to the sub reservoirs and sum them up
    for (center_x, center_y), (sigma_x, sigma_y), amplitude in zip(centers, sigmas, amplitudes):
        pressure_sources += gaussian_distribution(X, Y, center_x, center_y, sigma_x, sigma_y, amplitude)

    # Create a path object for the triangular zero zone
    zero_zone_path = Path(zero_zone_vertices)

    # Check each point in the grid to see if it is inside the zero zone or on its border
    for i in range(num_subs):
        for j in range(num_subs):
            if zero_zone_path.contains_point((j, i), radius=-1e-9):
                pressure_sources[i, j] = 0

    # Convert the 2D array to list
    pressure_list = pressure_sources.flatten().tolist()
    return pressure_list


# Test and visualization in the main function
if __name__ == '__main__':
    # 1) Global parameters: An example with 2 small regions
    num_subs = 125                  # 125 sub-reservoirs in the x or y directions. Total number is 125**2
    centers = [(20, 23), (75, 75)]  # 2 small regions
    sigmas = [(5, 8), (15, 18)]     # [(sigma_x1, sigma_y1), (sigma_x2, sigma_y2)]
    amplitudes = [10, 15]           # [amplitude1, amplitude2]

    # 2) Call the function
    pressure_list = assign_gaussian_pressure(num_subs, centers, sigmas, amplitudes)

    # 3) Visualization
    pressure_matrix = np.array(pressure_list).reshape((num_subs, num_subs))
    plt.imshow(pressure_matrix, extent=[0, num_subs, 0, num_subs], origin='lower', cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.scatter(*zip(*centers), color='red', marker='x')  # Mark the center of the Gaussian distribution
    plt.title('Pressure Distribution (Index Units)')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.tight_layout()
    plt.show()


