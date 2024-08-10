import numpy as np
import matplotlib.pyplot as plt
# Dimension reduction methods
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Plot samples after PCA
from matplotlib.patches import Ellipse
# Plot models
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def posterior_sampling(posterior_model_cov, posterior_model_mean, num_samples, random_seed=42):
    """
    Use Cholesky decomposition to sample models from the posterior model covariance matrix.
    :param posterior_model_cov: Posterior model covariance matrix; size = (M, M)
    :param posterior_model_mean: Posterior model mean; size = (M, 1)
    :param num_samples: Number of samples to generate
    :param random_seed: Random seed for reproducibility
    :return: Posterior model samples (size = (M, num_samples)) and distances from the mean.
    """
    np.random.seed(random_seed)
    # 1) Cholesky decomposition
    L = np.linalg.cholesky(posterior_model_cov)
    # 2) Generate random model samples
    n = len(posterior_model_cov)        # n = M
    model_samples = np.zeros((n, num_samples))  # size = (M, n_samples)
    distances = np.zeros(num_samples)
    # 3) Generate each sample in a loop
    for i in range(num_samples):
        z = np.random.randn(n)
        update_term = L @ z
        model_samples[:, i] = update_term + posterior_model_mean
        distances[i] = np.linalg.norm(update_term)
    print(f"Posterior {num_samples} model samples generated successfully!")
    return model_samples, distances


def plot_distances_histogram(distances, bins=50, file_name="Samples_Distances_Histogram.png"):
    """
    Plot a histogram of the distances.
    :param distances: Distances of the sampled models from the posterior model mean
    :param bins: Number of bins for the histogram
    :param file_name: If provided, save the plot to this file
    """
    # 1) Plot histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins, density=True, alpha=0.6, color='g', label='Sample Distances')

    # 2) Add labels and legend, and save
    plt.title('Histogram of Sample Distances from Posterior Mean')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


#############################################################################################################
########################################  PCA related functions  ############################################
#############################################################################################################
def pca_transformation(model_samples, n_components=2):
    """
    Perform PCA transformation and compute mean and covariance of the transformed samples.
    :param model_samples: The original high-dimensional model samples.
    :param n_components: The number of principal components.
    :return: PCA-transformed samples, mean of the transformed samples, covariance of the transformed samples, standard deviations.
    """
    pca = PCA(n_components=n_components)
    samples_pca = pca.fit_transform(model_samples.T)  # Transpose samples to fit PCA input shape
    mean_pca = np.mean(samples_pca, axis=0)
    cov_pca = np.cov(samples_pca, rowvar=False)
    std_dev_pca = np.sqrt(np.diag(cov_pca))
    return samples_pca, mean_pca, cov_pca, std_dev_pca


def filter_samples_within_custom_ranges(samples_pca, mean_pca, custom_std_ranges):
    """
    Filter samples within custom standardized ranges and return their indices.
    :param samples_pca: The PCA-transformed samples.
    :param mean_pca: The mean of the PCA-transformed samples.
    :param custom_std_ranges: A list of tuples representing the custom standard deviation ranges.
    :return: A list of arrays of indices of samples within each specified range.
    """
    distances = np.linalg.norm(samples_pca - mean_pca, axis=1)
    filtered_indices = []
    for lower, upper in custom_std_ranges:
        indices_in_range = np.where((distances > lower) & (distances <= upper))[0]
        filtered_indices.append(indices_in_range)
    return filtered_indices


def select_samples_within_custom_ranges(samples_pca, mean_pca, custom_std_ranges, num_selected=3, random_seed=42):
    """
    Obtain random samples (and indices) within custom standardized ranges.
    :param samples_pca: The PCA-transformed samples.
    :param mean_pca: The mean of the PCA-transformed samples.
    :param custom_std_ranges: A list of tuples representing the custom standard deviation ranges.
    :param num_selected: The number of samples to select from each range.
    :param random_seed: The random seed for reproducibility.
    :return: Randomly selected indices within each custom range.
    """
    filtered_indices = filter_samples_within_custom_ranges(samples_pca, mean_pca, custom_std_ranges)

    np.random.seed(random_seed)
    selected_indices = []

    for indices_in_range in filtered_indices:
        if len(indices_in_range) >= num_selected:
            selected_idx = np.random.choice(indices_in_range, num_selected, replace=False)
        else:
            selected_idx = indices_in_range

        selected_indices.append(selected_idx)

    return selected_indices



#############################################################################################################
####################################  Samples Display after PCA  ############################################
#############################################################################################################
def plot_ellipse(ax, mean, cov, n_std=1.0, **kwargs):
    """
    Plots an ellipse representing the covariance matrix.
    :param ax: The axis on which to plot.
    :param mean: The mean of the data.
    :param cov: The covariance matrix.
    :param n_std: The number of standard deviations.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    vx, vy = eigvecs[:, 0]
    theta = np.arctan2(vy, vx)
    theta = np.degrees(theta)

    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, **kwargs)
    ax.add_patch(ellipse)


def plot_pca_samples_with_ellipses(samples_pca, mean_pca, cov_pca, selected_indices_dict, file_name="posterior_samples_in_PCA_projection.png"):
    """
    Plot PCA samples with ellipses and selected points.
    :param samples_pca: The PCA-transformed samples.
    :param mean_pca: The mean of the PCA-transformed samples.
    :param cov_pca: The covariance of the PCA-transformed samples.
    :param selected_indices_dict: Dictionary containing label and indices of selected samples within different ranges.
    :param file_name: The file name to save the plot.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1) Plot all PCA-transformed samples
    plt.scatter(samples_pca[:, 0], samples_pca[:, 1], c='blue', label='Sampled Models', alpha=0.1)
    plt.scatter(mean_pca[0], mean_pca[1], c='red', label='Posterior Mean', marker='x', s=50)

    # 2) Add 1 std dev and 2 std dev ellipse
    plot_ellipse(ax, mean_pca, cov_pca, n_std=1.0, edgecolor='green', linestyle='--', linewidth=2, label='1 Std Dev')
    plot_ellipse(ax, mean_pca, cov_pca, n_std=2.0, edgecolor='orange', linestyle='--', linewidth=2, label='2 Std Dev')

    # 3) Highlight selected samples within different ranges
    colors = ['cyan', 'magenta', 'yellow', 'lime', 'purple', 'orange']
    for i, (label, indices) in enumerate(selected_indices_dict.items()):
        color = colors[i % len(colors)]  # cycle through colors if more labels than colors
        selected_samples = samples_pca[indices]
        plt.scatter(selected_samples[:, 0], selected_samples[:, 1], c=color, label=label, edgecolors='black', s=30)

    # 4) Add legend and labels
    plt.title(f'PCA 2D Projection of {samples_pca.shape[0]} Posterior Model Samples')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.axis('equal')

    # 5) Save and display
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


#############################################################################################################
#################################  Selected models (samples) Display  #######################################
#############################################################################################################
def plot_model_simplified(ax, R, num_subs, sub_pressures, show_xlabel=False, show_colorbar_label=False):
    """
    Plot the reservoir layer's horizontal profile.
    :param ax: The matplotlib axis to plot on
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param sub_pressures: The list of the sub-reservoirs pressures when num_subs != 0
    :param show_xlabel: Whether to show x-axis label
    :param show_colorbar_label: Whether to show colorbar label
    """
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    cmap = plt.get_cmap('jet')
    norm = Normalize()

    sub_pressures = [p / 1e6 for p in sub_pressures]  # Convert to MPa
    sub_radius = R / num_subs
    norm.autoscale(sub_pressures)
    for i in range(num_subs):
        for j in range(num_subs):
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
            ax.add_patch(circle)
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(sub_pressures)
    cbar = plt.colorbar(sm, ax=ax)
    if show_colorbar_label:
        cbar.set_label('Pressure [MPa]', fontsize=12)
    if show_xlabel:
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'Sub-reservoirs with radius {R/num_subs:.1f}m')


def plot_all_selected_models(R, num_subs, samples, selected_indices_dict, num_cols=3, file_name='selected_sampled_models_combined.png'):
    """
    Plot sampled models by custom standard deviation ranges.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param samples: The matrix of all samples
    :param selected_indices_dict: Dictionary containing range labels and their corresponding sample indices.
    :param num_cols: Number of columns in the subplot grid
    :param file_name: The name of the output file
    """
    num_ranges = len(selected_indices_dict)
    fig, axes = plt.subplots(num_ranges, num_cols, figsize=(15, 4 * num_ranges))
    axes = axes.flatten()

    # Helper function to plot a sample
    def plot_sample(ax, title, sample_index, show_xlabel, show_colorbar_label):
        sub_pressures = samples[:, sample_index]
        plot_model_simplified(ax=ax, R=R, num_subs=num_subs, sub_pressures=sub_pressures,
                              show_xlabel=show_xlabel, show_colorbar_label=show_colorbar_label)
        ax.set_title(title, fontsize=10)

    # Plot samples for each range
    idx = 0
    for range_label, sample_indices in selected_indices_dict.items():
        for i, index in enumerate(sample_indices[:num_cols]):
            show_xlabel = (idx >= num_cols * (num_ranges - 1))
            show_colorbar_label = (i % num_cols == num_cols - 1)
            plot_sample(axes[idx], f'Sample {i + 1} in {range_label}', int(index), show_xlabel, show_colorbar_label)
            idx += 1

    # Remove unused subplots
    for j in range(num_cols * num_ranges):
        if j >= idx:
            fig.delaxes(axes[j])

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()