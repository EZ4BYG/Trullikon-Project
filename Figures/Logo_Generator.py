import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal


def generate_grid(x_range, y_range, num_points):
    """Generates a 2D grid of points."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    return X, Y, pos


def generate_gaussian_distributions(pos, means, covariances):
    """Generates combined Gaussian distributions from given means and covariances."""
    Z_combined = np.zeros(pos.shape[:2])
    for mean, cov in zip(means, covariances):
        Z_combined += multivariate_normal(mean, cov).pdf(pos)
    return Z_combined


def metropolis_hastings(target_func, num_samples, initial_position, step_size):
    """Performs Metropolis-Hastings sampling."""
    samples = []
    current_position = initial_position
    current_prob = target_func(current_position)

    for _ in range(num_samples):
        proposed_position = current_position + np.random.normal(scale=step_size, size=2)
        proposed_prob = target_func(proposed_position)
        acceptance_prob = min(1, proposed_prob / current_prob)

        if np.random.rand() < acceptance_prob:
            current_position = proposed_position
            current_prob = proposed_prob

        samples.append(current_position)

    return np.array(samples)


def distance_to_nearest_peak(sample, means):
    """Calculates the distance from a sample to the nearest peak."""
    distances = [np.linalg.norm(sample - np.array(mean)) for mean in means]
    return min(distances)


def filter_samples(samples, means, distance_threshold):
    """Filters samples to keep those near the peaks."""
    return np.array([s for s in samples if distance_to_nearest_peak(s, means) < distance_threshold])


def plot_results(X, Y, Z_combined, filtered_samples, output_file):
    """Plots the results and saves the plot as an SVG file."""
    plt.figure(figsize=(8, 8))
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Draw contour plot for the combined Gaussian distribution with three peaks
    plt.contour(X, Y, Z_combined, levels=7, cmap="jet", linewidths=2)
    # Plot the filtered sampled points
    plt.scatter(filtered_samples[:, 0], filtered_samples[:, 1], color='k', s=12, alpha=0.6)

    # Add text labels
    plt.text(x=4, y=0.5, s='Trullikon Project', fontsize=50, ha='left', va='center')
    plt.text(x=4.2, y=-0.5, s='A Toolbox for Data Analysis, Forward Model Simulation,\nBayesian Inversion and Posterior Sampling in\nA real CO2 injection site located in Switzerland, Trullikon-1-1', fontsize=15, ha='left', va='center')

    # Set limits to make sure the plot is centered
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    # Remove all axis labels and ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Save as SVG with transparent background
    plt.savefig(output_file, format='svg', transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    num_samples = 2000
    step_size = 0.5

    # Define parameters for the Gaussian distributions
    means = [[-2, -2], [2, 0], [0, 1.9]]
    covariances = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0.3], [0.3, 1]]]

    # Generate the grid and the combined Gaussian distributions
    X, Y, pos = generate_grid(x_range=(-5, 5), y_range=(-5, 5), num_points=500)
    Z_combined = generate_gaussian_distributions(pos, means, covariances)

    # Define the target function for sampling directly using the means and covariances
    def target_func(xy):
        return sum(multivariate_normal(mean, cov).pdf(xy) for mean, cov in zip(means, covariances))

    # Perform Metropolis-Hastings sampling
    initial_position = np.array([0, 0])
    samples = metropolis_hastings(target_func=target_func, num_samples=num_samples, initial_position=initial_position,
                                  step_size=step_size)

    # Filter the samples to keep those near the peaks
    filtered_samples = filter_samples(samples, means, distance_threshold=1.8)

    # Plot and save the results
    plot_results(X, Y, Z_combined, filtered_samples, "Three_Peaks_Logo.svg")