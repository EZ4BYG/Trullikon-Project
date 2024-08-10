import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import truncnorm


def chi(model, model_prior, Cd, Cm, G, dobs):
    """
    Calculate the misfit value of the inversion.
    :param model: The model vector
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    """
    # 1) Calculate the misfit value
    Cm_inv = np.linalg.inv(Cm)
    Cd_inv = np.linalg.inv(Cd)
    term1 = 0.5 * (model - model_prior).T @ Cm_inv @ (model - model_prior)
    term2 = 0.5 * (G @ model - dobs).T @ Cd_inv @ (G @ model - dobs)
    misfit = term1 + term2
    return misfit


def metropolis_hastings_with_nonnegative_projection(initial_sample, num_samples, initial_proposal_std, max_value,
                                                    model_prior, Cd, Cm, G, dobs):
    """
    Metropolis-Hastings algorithm with non-negative projection and adaptive proposal standard deviation.
    :param initial_sample: The initial sample, which is the solution of BNNLSM
    :param num_samples: The number of samples to generate
    :param initial_proposal_std: The initial standard deviation of the Gaussian proposal distribution
    :param max_value: The maximum value of the model vector
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    :return: The generated samples matrix
    """
    # 1) Initialize
    samples = []
    current_sample = initial_sample
    proposal_std = initial_proposal_std
    acceptance_rate_target = 0.2              # Target acceptance rate
    adaptation_interval = num_samples // 100  # Automatically adjust the proposal standard deviation every 1000 samples
    acceptance_rates = []

    # 2) Generate samples
    for i in range(num_samples):
        # 2.1) Generate a proposal sample with non-negative projection
        proposal_sample = current_sample + np.random.normal(0, proposal_std, size=current_sample.shape)
        proposal_sample = np.maximum(proposal_sample, 0)          # non-negative projection
        proposal_sample = np.minimum(proposal_sample, max_value)  # upper-boundary constraint

        # 2.2) Calculate the misfit value of the current and proposal samples
        current_chi = chi(model=current_sample, model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
        proposal_chi = chi(model=proposal_sample, model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
        compared_term = np.exp(current_chi - proposal_chi)

        # 2.3) Calculate the acceptance ratio
        acceptance_ratio = min(1, compared_term)

        # 2.4) Accept or reject the proposal sample
        if np.random.rand() < acceptance_ratio:
            current_sample = proposal_sample
            samples.append(current_sample)
            acceptance_rates.append(1)
        else:
            acceptance_rates.append(0)

        # 2.5) Adaptive adjustment of the proposal standard deviation
        if (i + 1) % adaptation_interval == 0:
            actual_acceptance_rate = np.mean(acceptance_rates[-adaptation_interval:])
            accepted_samples_count = np.sum(acceptance_rates[-adaptation_interval:])
            print(f"Interval {i // adaptation_interval + 1}: Proposal Std: {proposal_std:.4f}, "
                  f"Accepted Samples: {accepted_samples_count}, Acceptance Rate: {actual_acceptance_rate:.2f}")

            if actual_acceptance_rate < acceptance_rate_target:
                proposal_std *= 0.95
            else:
                proposal_std *= 1.05

    return np.array(samples)


#############################################################################################################
###############################  Posterior Analysis: calculate chi values  ##################################
#############################################################################################################
# Viewpoint 1: Chi values difference
def amha_posterior_analysis_chi(amha_accepted_samples, initial_sample, bins, model_prior, Cd, Cm, G, dobs,
                                trim=50, file_name="amha_posterior_chi_values_distribution.png"):
    """
    Posterior analysis: calculate chi values for the accepted samples and sort + display them.
    :param amha_accepted_samples: The accepted samples from the Adaptive Metropolis-Hastings algorithm (only change this)
    :param initial_sample: The initial sample, which is the solution of BNNLSM
    :param bins: The number of bins for the histogram
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    :param trim: The number of samples to be trimmed on both sides
    :param file_name: The name of the output file
    return: The binned trimmed sorted samples
    """
    # 1) Calculate chi values for the accepted samples
    print("Calculating chi values for the accepted samples...")
    chi_values = np.array([chi(sample, model_prior, Cd, Cm, G, dobs) for sample in amha_accepted_samples])
    chi_value_initial_sample = chi(initial_sample, model_prior, Cd, Cm, G, dobs)
    # 2) Sort original accepted samples based on chi values: descending order
    sorted_indices = np.argsort(-chi_values)
    sorted_samples = amha_accepted_samples[sorted_indices]
    sorted_chi_values = chi_values[sorted_indices]
    # 3) Trim the sorted chi values
    trimmed_sorted_chi_values = sorted_chi_values[trim:-trim]
    # trimmed_sorted_samples = sorted_samples[trim:-trim]

    # 4) Fit a truncated normal distribution to the trimmed chi values
    # a: lower bound, b: upper bound
    # mu: mean, sigma: standard deviation
    a, b = (min(trimmed_sorted_chi_values) - np.mean(trimmed_sorted_chi_values)) / np.std(trimmed_sorted_chi_values), (
            max(trimmed_sorted_chi_values) - np.mean(trimmed_sorted_chi_values)) / np.std(trimmed_sorted_chi_values)
    mu, sigma = np.mean(trimmed_sorted_chi_values), np.std(trimmed_sorted_chi_values)
    fitted_dist = truncnorm(a, b, loc=mu, scale=sigma)

    # 5) Display the trimmed sorted chi values
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 5.1) Histogram: trimmed sorted chi values
    _, bins_edges, _ = ax1.hist(trimmed_sorted_chi_values, bins=bins, alpha=0.7, color='b', label='Histogram')
    ax1.set_xlabel('Chi value', fontsize=12)
    ax1.set_ylabel('Count', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    # 5.2) Truncated Gaussian Fit
    ax2 = ax1.twinx()
    x = np.linspace(min(trimmed_sorted_chi_values), max(trimmed_sorted_chi_values), 1000)
    ax2.plot(x, fitted_dist.pdf(x), label='Truncated Gaussian Fit', color='g')
    ax2.set_ylabel('Density', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, max(fitted_dist.pdf(x)) * 1.1)
    # 5.3) Display settings
    title1 = f"HMC: Distribution of trimmed Chi values. Chi value of the initial sample is {chi_value_initial_sample:.2f}\n"
    title2 = f"Trimmed {trim} points from both ends out of {len(chi_values)} total accepted samples\n"
    title3 = f"Truncated Gaussian Fit: MLE = {mu:.2f}, SD = {sigma:.2f}"
    plt.title(title1 + title2 + title3, fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print("Chi values have been calculated and sorted successfully!")

    # 6) Return the sorted original samples
    return sorted_samples


# Viewpoint 2: Euclidean distance
def amha_posterior_analysis_distance(amha_accepted_samples, initial_sample, bins, trim=50,
                                     file_name="amha_posterior_distances_distribution.png"):
    """
    Posterior analysis: calculate Euclidean distances between the accepted samples and the initial sample, sort + display them.
    :param amha_accepted_samples: The accepted samples from the HMC algorithm
    :param initial_sample: The initial sample (the solution of BNNLSM or BBLSM)
    :param bins: The number of bins for the histogram
    :param trim: The number of samples to be trimmed on both sides
    :param file_name: The name of the output file
    :return: The binned trimmed sorted samples based on Euclidean distance
    """
    # 1) Calculate Euclidean distances between accepted samples and the initial sample
    print("Calculating Euclidean distances between the accepted samples and the initial sample...")
    distances = np.linalg.norm(amha_accepted_samples - initial_sample, axis=1)

    # 2) Sort the samples based on the distances: ascending order
    sorted_indices = np.argsort(distances)
    sorted_samples = amha_accepted_samples[sorted_indices]
    sorted_distances = distances[sorted_indices]

    # 3) Trim the sorted distances
    trimmed_sorted_distances = sorted_distances[trim:-trim]
    # trimmed_sorted_samples = sorted_samples[trim:-trim]  # Uncomment if you need to return trimmed samples

    # 4) Display the trimmed sorted distances
    plt.figure(figsize=(10, 6))
    plt.hist(trimmed_sorted_distances, bins=bins, alpha=0.7, color='b')
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    title1 = f"HMC: Distribution of trimmed Euclidean distances from the initial sample\n"
    title2 = f"Trimmed {trim} points from both ends out of {len(distances)} total accepted samples"
    plt.title(title1 + title2, fontsize=14)

    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print("Euclidean distances have been calculated and sorted successfully!")

    # 5) Return the sorted original samples
    return sorted_samples


#############################################################################################################
#################################  Selected models (samples) Display  #######################################
#############################################################################################################
def plot_each_model(ax, R, num_subs, sub_pressures, show_xlabel=False, show_colorbar_label=False):
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


def plot_all_selected_mh_models(R, num_subs, samples, selected_indices, num_cols=3, file_name='selected_sampled_mh_models_combined.png'):
    """
    Plot sampled (Metropolis_Hastings, simplified as 'mh') models based on selected indices.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param samples: The matrix of all samples
    :param selected_indices: List or array of selected sample indices to plot
    :param num_cols: Number of columns in the subplot grid
    :param file_name: The name of the output file
    """
    num_samples = len(selected_indices)
    num_rows = (num_samples + num_cols - 1) // num_cols  # Calculate number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()

    # Helper function to plot a sample
    def plot_sample(ax, title, sample_index, show_xlabel, show_colorbar_label):
        sub_pressures = samples[sample_index]
        plot_each_model(ax=ax, R=R, num_subs=num_subs, sub_pressures=sub_pressures,
                        show_xlabel=show_xlabel, show_colorbar_label=show_colorbar_label)
        ax.set_title(title, fontsize=10)

    # Plot samples
    for idx, sample_index in enumerate(selected_indices):
        show_xlabel = (idx >= num_cols * (num_rows - 1))
        show_colorbar_label = (idx % num_cols == num_cols - 1)
        plot_sample(axes[idx], f'Sample {sample_index + 1}', sample_index, show_xlabel, show_colorbar_label)

    # Remove unused subplots
    for j in range(num_cols * num_rows):
        if j >= num_samples:
            fig.delaxes(axes[j])

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()