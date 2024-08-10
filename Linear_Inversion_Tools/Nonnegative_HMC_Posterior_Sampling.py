import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import truncnorm


#############################################################################################################
#################################  Potential Energy U(m) related terms  #####################################
#############################################################################################################
def chi(model, model_prior, Cd, Cm, G, dobs):
    """
    Calculate the misfit function value of the linear inversion: Potential Energy U(m).
    :param model: The current model vector (only this changes everytime)
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    """
    Cm_inv = np.linalg.inv(Cm)
    Cd_inv = np.linalg.inv(Cd)
    term1 = 0.5 * (model - model_prior).T @ Cm_inv @ (model - model_prior)
    term2 = 0.5 * (G @ model - dobs).T @ Cd_inv @ (G @ model - dobs)
    misfit = term1 + term2
    return misfit


def gradient_chi(model, model_prior, Cd, Cm, G, dobs):
    """
    Calculate the deterministic first-order gradient of the misfit function value of the linear inversion: first-order gradient of Potential Energy.
    :param model: The current model vector (only this changes everytime)
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    """
    Cm_inv = np.linalg.inv(Cm)
    Cd_inv = np.linalg.inv(Cd)
    grad_term1 = Cm_inv @ (model - model_prior)
    grad_term2 = G.T @ Cd_inv @ (G @ model - dobs)
    gradient = grad_term1 + grad_term2
    return gradient


def hessian_chi(Cd, Cm, G):
    """
    Calculate the deterministic second-order (Hessian matrix) of the misfit function value of the linear inversion: second-order gradient of Potential Energy.
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    """
    Cm_inv = np.linalg.inv(Cm)
    Cd_inv = np.linalg.inv(Cd)
    hessian = Cm_inv + G.T @ Cd_inv @ G
    return hessian


#############################################################################################################
#################################  Kinetic Energy K(p) related terms  #######################################
#############################################################################################################
def kinetic_energy(momentum, mass_matrix_inv):
    """
    Calculate the kinetic energy K(p) of the system.
    """
    return 0.5 * momentum.T @ mass_matrix_inv @ momentum


def gradient_kinetic_energy(momentum, mass_matrix_inv):
    """
    Calculate the gradient of the kinetic energy K(p) of the system.
    """
    return mass_matrix_inv @ momentum


#############################################################################################################
#################################  Hamiltonian Monte Carlo Functions  #######################################
#############################################################################################################
def leapfrog_integration(m_new, p_new, mass_matrix_inv, step_size, leapfrog_steps, max_value,
                         model_prior, Cd, Cm, G, dobs):
    """
    Leapfrog integration for Hamiltonian Monte Carlo.
    :param m_new: The current model vector (only this changes everytime)
    :param p_new: The current momentum vector (only this changes everytime)
    :param mass_matrix_inv: The inverse of the mass matrix
    :param step_size: The step size of the leapfrog steps -> exploration step size
    :param leapfrog_steps: The number of leapfrog steps -> exploration number
    :param max_value: The maximum value for the non-negative projection (upper boundary)
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    """
    # Half-step update momentum
    p_new = p_new - 0.5 * step_size * gradient_chi(model=m_new, model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
    # Leapfrog steps: exploration
    for l in range(leapfrog_steps):
        # Full-step update model + non-negative projection
        m_new = m_new + step_size * gradient_kinetic_energy(p_new, mass_matrix_inv)
        m_new = np.maximum(m_new, 0)          # non-negative constraint
        m_new = np.minimum(m_new, max_value)  # upper-boundary constraint
        # Full-step update momentum
        if l < leapfrog_steps - 1:
            p_new = p_new - step_size * gradient_chi(m_new, model_prior, Cd, Cm, G, dobs)
    # Half-step update momentum again
    p_new = p_new - 0.5 * step_size * gradient_chi(m_new, model_prior, Cd, Cm, G, dobs)
    # Reverse momentum
    p_new = -p_new
    return m_new, p_new


def hmc_with_nonnegative_projection(initial_sample, num_samples, leapfrog_steps, step_size, max_value,
                                    model_prior, Cd, Cm, G, dobs):
    """
    Hamiltonian Monte Carlo algorithm with non-negative projection.
    :param initial_sample: The starting model vector (the solution of BNNLSM or BBLSM)
    :param num_samples: The number of samples
    :param leapfrog_steps: The number of leapfrog steps -> exploration number
    :param step_size: The step size of the leapfrog steps -> exploration step size
    :param max_value: The maximum value for the non-negative projection (upper boundary)
    :param model_prior: The prior model vector
    :param Cd: The data covariance matrix
    :param Cm: The model covariance matrix
    :param G: The sensitivity matrix
    :param dobs: The observed data vector
    """
    # 0) Initialization
    samples = []
    # Mass matrix option 1: Hessian matrix
    # M = hessian_chi(Cd=Cd, Cm=Cm, G=G)
    # Mass matrix option 2: Diagonal matrix based on Cm
    M = 0.5 *np.diag(np.linalg.inv(Cm).diagonal())
    M_inv = np.linalg.inv(M)
    current_sample = initial_sample

    # 1) HMC iteration
    print("HMC with non-negative projection posterior sampling started...")
    for i in range(num_samples):
        # 1) Initialize the current sample (position m) and momentum (p)
        current_momentum = np.random.multivariate_normal(np.zeros_like(initial_sample), M)
        # 1.1) Current Hamiltonian
        current_potential_energy = chi(model=current_sample, model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
        current_kinetic_energy = kinetic_energy(momentum=current_momentum, mass_matrix_inv=M_inv)
        current_total_energy = current_potential_energy + current_kinetic_energy

        # 2) Leapfrog: approximate the Hamiltonian dynamics two-coupled functions
        proposal_sample = current_sample
        proposal_momentum = current_momentum
        proposal_sample, proposal_momentum = leapfrog_integration(m_new=proposal_sample, p_new=proposal_momentum, mass_matrix_inv=M_inv,
                                                                  step_size=step_size, leapfrog_steps=leapfrog_steps, max_value=max_value,
                                                                  model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
        # 2.1) Proposal Hamiltonian
        proposal_potential_energy = chi(model=proposal_sample, model_prior=model_prior, Cd=Cd, Cm=Cm, G=G, dobs=dobs)
        proposal_kinetic_energy = kinetic_energy(momentum=proposal_momentum, mass_matrix_inv=M_inv)
        proposal_total_energy = proposal_potential_energy + proposal_kinetic_energy
        print(f"Sample {i + 1}/{num_samples}: Current total energy = {current_total_energy:.2f}, Proposal total energy = {proposal_total_energy:.2f}")

        # 3.2) Acceptance ratio
        acceptance_ratio = np.exp(-(proposal_total_energy - current_total_energy))
        print(f"Sample {i + 1}/{num_samples}: Acceptance ratio = {acceptance_ratio:.2f}")
        if np.random.rand() <= acceptance_ratio:
            current_sample = proposal_sample
            # Only record the accepted samples
            samples.append(current_sample)

    # Print information finally
    total_accepted_samples = len(samples)
    acceptance_rate = total_accepted_samples / num_samples
    print("HMC with non-negative projection posterior sampling finished successfully!")
    print(f"Total sample number is {num_samples}. Total accepted sample number is {total_accepted_samples}. Acceptance rate is {acceptance_rate:.2f}")

    return np.array(samples)


#############################################################################################################
###############################  Posterior Analysis: calculate chi values  ##################################
#############################################################################################################
# Viewpoint 1: Chi values difference
def hmc_posterior_analysis_chi(hmc_accepted_samples, initial_sample, bins, model_prior, Cd, Cm, G, dobs,
                               trim=100, file_name="hmc_posterior_chi_values_distribution.png"):
    """
    Posterior analysis: calculate chi values for the accepted samples and sort + display them.
    :param hmc_accepted_samples: The accepted samples from the HMC algorithm (only change this)
    :param initial_sample: The initial sample (the solution of BNNLSM or BBLSM)
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
    chi_values = np.array([chi(sample, model_prior, Cd, Cm, G, dobs) for sample in hmc_accepted_samples])
    chi_value_initial_sample = chi(initial_sample, model_prior, Cd, Cm, G, dobs)
    # 2) Sort original accepted samples based on chi values: descending order
    sorted_indices = np.argsort(-chi_values)
    sorted_samples = hmc_accepted_samples[sorted_indices]
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


# Viewpoint 2: Euclidean distances
def hmc_posterior_analysis_distance(hmc_accepted_samples, initial_sample, bins, trim=100,
                                    file_name="hmc_posterior_distances_distribution.png"):
    """
    Posterior analysis: calculate Euclidean distances between the accepted samples and the initial sample,
    sort + display them.
    :param hmc_accepted_samples: The accepted samples from the HMC algorithm
    :param initial_sample: The initial sample (the solution of BNNLSM or BBLSM)
    :param bins: The number of bins for the histogram
    :param trim: The number of samples to be trimmed on both sides
    :param file_name: The name of the output file
    :return: The binned trimmed sorted samples based on Euclidean distance
    """
    # 1) Calculate Euclidean distances between accepted samples and the initial sample
    print("Calculating Euclidean distances between the accepted samples and the initial sample...")
    distances = np.linalg.norm(hmc_accepted_samples - initial_sample, axis=1)

    # 2) Sort the samples based on the distances: ascending order
    sorted_indices = np.argsort(distances)
    sorted_samples = hmc_accepted_samples[sorted_indices]
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


def plot_all_selected_hmc_models(R, num_subs, samples, selected_indices, num_cols=3, file_name='selected_sampled_mh_models_combined.png'):
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