import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, multivariate_normal


def plot_data_obs_pred(X, Y, data_obs, data_pred, data_type, file_name="data_obs_pred_comparison.png"):
    """
    Plot the comparison of observed and predicted data.
    :param X: x-coordinates
    :param Y: y-coordinates
    :param data_obs: observed data
    :param data_pred: predicted data
    :param data_type: the type of the data
    :param file_name: the file name to save the plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot observed data
    im0 = ax[0].imshow(data_obs, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='viridis')
    ax[0].set_title(f'Observed Data ({data_type})')
    ax[0].set_xlabel('X [m]')
    ax[0].set_ylabel('Y [m]')
    ax[0].grid()
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.2)
    cbar0 = plt.colorbar(im0, cax=cax0)
    cbar0.set_label(f'Observed {data_type}')

    # Plot predicted data
    im1 = ax[1].imshow(data_pred, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='viridis')
    ax[1].set_title(f'Predicted Data ({data_type})')
    ax[1].set_xlabel('X [m]')
    ax[1].set_ylabel('Y [m]')
    ax[1].grid()
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label(f'Predicted {data_type}')

    plt.tight_layout()
    file_name = data_type + "_" + file_name
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_prior_est(R, num_subs, model_prior, model_est, model_noise_deviation, file_name="model_prior_est_comparison.png"):
    """
    Plot the comparison of the prior and estimated models.
    :param R: The radius of the whole reservoir
    :param num_subs: The number of sub-reservoirs in the x or y direction
    :param model_prior: The prior model
    :param model_est: The estimated model (inversion result)
    :param model_noise_deviation: The standard deviation of the prior model
    :param file_name: the file name to save the plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    sub_radius = R / num_subs
    cmap = plt.get_cmap('jet')

    # Plot prior model
    norm_prior = Normalize()
    model_prior = model_prior / 1e6  # Convert to MPa
    norm_prior.autoscale(model_prior)
    for i in range(num_subs):
        for j in range(num_subs):
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm_prior(model_prior[index])), ec='black')
            ax[0].add_patch(circle)
    ax[0].set_xlim(-R, R)
    ax[0].set_ylim(-R, R)
    ax[0].set_xlabel('X [m]', fontsize=12)
    ax[0].set_ylabel('Y [m]', fontsize=12)
    ax[0].set_title('Prior Model', fontsize=14)
    sm_prior = ScalarMappable(cmap=cmap, norm=norm_prior)
    sm_prior.set_array(model_prior)
    cbar_prior = plt.colorbar(sm_prior, ax=ax[0])
    cbar_prior.set_label('Pressure Changes [MPa]', fontsize=12)

    # Plot estimated model
    norm_est = Normalize()
    model_est = model_est / 1e6  # Convert to MPa
    norm_est.autoscale(model_est)
    for i in range(num_subs):
        for j in range(num_subs):
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm_est(model_est[index])), ec='black')
            ax[1].add_patch(circle)
    ax[1].set_xlim(-R, R)
    ax[1].set_ylim(-R, R)
    ax[1].set_xlabel('X [m]', fontsize=12)
    ax[1].set_ylabel('Y [m]', fontsize=12)
    ax[1].set_title('Estimated Model', fontsize=14)
    sm_est = ScalarMappable(cmap=cmap, norm=norm_est)
    sm_est.set_array(model_est)
    cbar_est = plt.colorbar(sm_est, ax=ax[1])
    cbar_est.set_label('Pressure Changes [MPa]', fontsize=12)

    fig.suptitle(
        f'Horizontal Profile of the Reservoir Model Layer: Prior Model vs Estimated Model\n'
        f'This Reservoir Model Has {num_subs ** 2} Sub-Reservoirs, Each with a Radius of {R / num_subs:.1f}m\n'
        f'The Prior Model Parameters Are Uncorrelated with Uniform Variance {model_noise_deviation / 1e6}MPa',
        fontsize=16)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


#######################################################################################################
#########################     Prior and Posterior Models Display functions     ########################
#######################################################################################################
def plot_prior_model(R, num_subs, prior_model, title="Prior Reservoir Model: Mean Values", file_name="prior_model_matrix.png"):
    """
    PLot the reservoir layer's horizontal profile.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param prior_model: The 1d vector of the prior model
    :param title: The title of the plot
    :param file_name: The file name to save the plot
    """
    # 1) Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    cmap = plt.get_cmap('jet')
    norm = Normalize()

    # 2) Plot the posterior model
    sub_pressures = [p / 1e6 for p in prior_model]  # Convert to MPa
    sub_radius = R / num_subs
    norm.autoscale(sub_pressures)
    for i in range(num_subs):
        for j in range(num_subs):
            # Note there: row-major order! so x uses j, y uses i.
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
            ax.add_patch(circle)
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(sub_pressures)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Pressure [MPa]', fontsize=12)
    title = title + f"\n{num_subs ** 2} sub-reservoirs, the radius of each sub-reservoir is {R / num_subs:.1f}m"
    ax.set_title(title, fontsize=14)

    # 3) Save and show
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_posterior_model(R, num_subs, posterior_model, title="Posterior Reservoir Model", file_name="posterior_model_matrix.png"):
    """
    PLot the reservoir layer's horizontal profile.
    :param R: The radius of the reservoir
    :param num_subs: The total number of sub-reservoirs equally divided in the x or y direction
    :param posterior_model: The 1d vector of the posterior model
    :param title: The title of the plot
    :param file_name: The file name to save the plot
    """
    # 1) Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    cmap = plt.get_cmap('jet')
    norm = Normalize()

    # 2) Plot the posterior model
    sub_pressures = [p / 1e6 for p in posterior_model]  # Convert to MPa
    sub_radius = R / num_subs
    norm.autoscale(sub_pressures)
    for i in range(num_subs):
        for j in range(num_subs):
            # Note there: row-major order! so x uses j, y uses i.
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            circle = patches.Circle((x, y), sub_radius, color=cmap(norm(sub_pressures[index])), ec='black')
            ax.add_patch(circle)
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(sub_pressures)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Pressure [MPa]', fontsize=12)
    title = title + f"\n{num_subs ** 2} sub-reservoirs, the radius of each sub-reservoir is {R / num_subs:.1f}m"
    ax.set_title(title, fontsize=14)

    # 3) Save and show
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


#######################################################################################################
#########################    Prior and Posterior Model Uncertainty Displays   #########################
#######################################################################################################
def plot_posterior_and_uncertainty(R, num_subs, prior_model, posterior_model, prior_model_cov, posterior_model_cov, selected_model_indices,
                                   title_right="Posterior Reservoir Model", file_name="posterior_model_uncertainty_analysis.png"):
    """
    Plot the posterior model and uncertainty distributions for selected indices.
    :param R: Radius of the reservoir
    :param num_subs: Number of sub-reservoirs in each direction
    :param prior_model: 1D vector of the prior model
    :param posterior_model: 1D vector of the posterior model
    :param prior_model_cov: 2D matrix of prior model covariance
    :param posterior_model_cov: 2D matrix of posterior model covariance
    :param selected_model_indices: Dict of indices with additional display info (color and label)
    :param title_right: Title of the right panel
    :param file_name: File name to save the plot
    """
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], height_ratios=[1, 1, 1])

    # 1) Right panel: the posterior model with 3 selected model parameters
    # 1.1) Set up the right panel
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_xlim(-R, R)
    ax2.set_ylim(-R, R)
    ax2.set_title(title_right)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    cmap = plt.get_cmap('jet')
    posterior_model_for_plot = [p / 1e6 for p in posterior_model]
    norm_plot = Normalize(vmin=np.min(posterior_model_for_plot), vmax=np.max(posterior_model_for_plot))
    # 1.2) Plot the posterior model
    sub_radius = R / num_subs
    for i in range(num_subs):
        for j in range(num_subs):
            x = -R + (2 * j + 1) * sub_radius
            y = -R + (2 * i + 1) * sub_radius
            index = i * num_subs + j
            color = cmap(norm_plot(posterior_model_for_plot[index]))
            circle = patches.Circle((x, y), sub_radius, color=color, ec='black')
            ax2.add_patch(circle)
            if index in selected_model_indices:
                rect = patches.Rectangle((x - sub_radius, y - sub_radius), 2 * sub_radius, 2 * sub_radius,
                                         linewidth=2, edgecolor=selected_model_indices[index]['color'], facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x, y, selected_model_indices[index]['label'], color='white', ha='center', va='center')

    sm = ScalarMappable(cmap=cmap, norm=norm_plot)
    sm.set_array(posterior_model_for_plot)
    plt.colorbar(sm, ax=ax2, orientation='vertical', label='Pressure Changes [MPa]')

    # 2) Left panel: Loop plot prior and posterior uncertainty distributions for selected indices (model parameters)
    for idx, (index, info) in enumerate(selected_model_indices.items()):
        ax = fig.add_subplot(gs[idx, 0])
        # 2.1) Get necessary info: convert to MPa
        mean_prior = prior_model[index] / 1e6
        std_prior = np.sqrt(prior_model_cov[index, index]) / 1e6
        mean_posterior = posterior_model_for_plot[index]  # Note: already in MPa before
        std_posterior = np.sqrt(posterior_model_cov[index, index]) / 1e6
        # 2.2) Set the range of x-axis
        x_min = min(mean_prior - 3 * std_prior, mean_posterior - 3 * std_posterior)
        x_max = max(mean_prior + 3 * std_prior, mean_posterior + 3 * std_posterior)
        x = np.linspace(x_min, x_max, 400)
        # 2.3) Compute the prior and posterior pdf
        prior_pdf = norm.pdf(x, mean_prior, std_prior)
        posterior_pdf = norm.pdf(x, mean_posterior, std_posterior)
        # 2.4) Plot the prior and posterior pdf
        # 2.4.1) PDFs: lines + fill_between
        ax.plot(x, prior_pdf, 'k-', label=f'Prior (μ={mean_prior:.2f}, σ={std_prior:.2f})')
        ax.fill_between(x, prior_pdf, color='grey', alpha=0.3)
        ax.plot(x, posterior_pdf, color=info['color'], linestyle='-', label=f'Posterior (μ={mean_posterior:.2f}, σ={std_posterior:.2f})')
        ax.fill_between(x, posterior_pdf, color=info['color'], alpha=0.3)
        # 2.4.2) Vertical mean lines
        y_pdf_max = max(ax.get_ylim())
        prior_ymax = norm.pdf(mean_prior, mean_prior, std_prior) / y_pdf_max
        posterior_ymax = norm.pdf(mean_posterior, mean_posterior, std_posterior) / y_pdf_max
        ax.axvline(mean_prior, color='k', linestyle='--', ymax=prior_ymax)
        ax.axvline(mean_posterior, color=info['color'], linestyle='--', ymax=posterior_ymax)
        # 2.4.3) Set the title, labels, and legend
        ax.set_title(f'Parameter {index} Uncertainty Distribution')
        ax.set_xlabel('Pressure Changes [MPa]')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


#######################################################################################################
##################  Prior and Posterior 2 model parameters Joint and Marginal PDF  ####################
#######################################################################################################
def compute_correlation(cov_matrix, index1, index2):
    """
    Compute the correlation between two model parameters.
    :param cov_matrix: The covariance matrix of the model parameters
    :param index1: The index of the first model parameter
    :param index2: The index of the second model parameter
    """
    cov_ij = cov_matrix[index1, index2]
    var_i = cov_matrix[index1, index1]
    var_j = cov_matrix[index2, index2]
    correlation = cov_ij / np.sqrt(var_i * var_j)
    return correlation


def plot_joint_and_marginals(ax_joint, ax_marg_x, ax_marg_y, mean, cov, index1, index2, type):
    """
    Plot the joint and marginal distributions of two model parameters.
    :param ax_joint: The joint distribution plot (controlled axis 'ax_joint')
    :param ax_marg_x: The marginal distribution plot for the first parameter (controlled axis 'ax_marg_x')
    :param ax_marg_y: The marginal distribution plot for the second parameter (controlled axis 'ax_marg_y')
    :param mean: The original mean vector of the model parameters, size = (M, 1)
    :param cov: The original covariance matrix of the model parameters, size = (M, M)
    :param index1: The index of the first model parameter
    :param index2: The index of the second model parameter
    :param type: The type of the model parameters, only for the title: 'Prior' or 'Posterior'
    """
    # 1) Select the mean and covariance (sub matrix) of the two model parameters
    selected_mean = mean[[index1, index2]] / 1e6
    selected_cov = cov[np.ix_([index1, index2], [index1, index2])] / (1e6 ** 2)  # Convert to MPa^2
    std_dev = np.sqrt(np.diag(selected_cov))  # Already in [MPa]
    print(selected_cov)

    # 2) Plot the joint distributions
    # 2.1) Set the reasonable range for the joint distribution
    x_min_joint, x_max_joint = selected_mean[0] - 3 * std_dev[0], selected_mean[0] + 3 * std_dev[0]
    y_min_joint, y_max_joint = selected_mean[1] - 3 * std_dev[1], selected_mean[1] + 3 * std_dev[1]
    x, y = np.mgrid[x_min_joint:x_max_joint:0.01, y_min_joint:y_max_joint:0.01]
    pos = np.dstack((x, y))
    # 2.2) Plot the joint distribution
    correlation = compute_correlation(cov, index1, index2)
    rv = multivariate_normal(selected_mean, selected_cov)
    ax_joint.contourf(x, y, rv.pdf(pos), levels=40, cmap='viridis')
    ax_joint.set_xlabel(f'Model Parameter {index1} [MPa]')
    ax_joint.set_ylabel(f'Model Parameter {index2} [MPa]')
    ax_joint.set_title(f'Joint Distribution of $m_{{{index1}}}$ and $m_{{{index2}}}$ ({type}). Correlation = {correlation:.4f}')
    ax_joint.grid(True)

    # 4) Plot marginal distribution for the first parameter
    x_marginal = np.linspace(x_min_joint, x_max_joint, 400)
    ax_marg_x.plot(x_marginal, norm.pdf(x_marginal, selected_mean[0], std_dev[0]), 'k-')
    ax_marg_x.axvline(selected_mean[0], color='r', linestyle='--', label=f'μ = {selected_mean[0]:.2f}')
    ax_marg_x.axvline(selected_mean[0] + std_dev[0], color='g', linestyle='--', label=f'σ = {std_dev[0]:.2f}')
    ax_marg_x.axvline(selected_mean[0] - std_dev[0], color='g', linestyle='--')
    ax_marg_x.set_title(r'marginal $p(m_{' + str(index1) + '})$')  # Adding a title
    ax_marg_x.xaxis.tick_top()
    ax_marg_x.xaxis.set_label_position('top')
    ax_marg_x.legend()

    # 5) Plot marginal distribution for the second parameter
    y_marginal = np.linspace(y_min_joint, y_max_joint, 400)
    ax_marg_y.plot(norm.pdf(y_marginal, selected_mean[1], std_dev[1]), y_marginal, 'k-')
    ax_marg_y.axhline(selected_mean[1], color='r', linestyle='--', label=f'μ = {selected_mean[1]:.2f}')
    ax_marg_y.axhline(selected_mean[1] + std_dev[1], color='g', linestyle='--', label=f'σ = {std_dev[1]:.2f}')
    ax_marg_y.axhline(selected_mean[1] - std_dev[1], color='g', linestyle='--')
    ax_marg_y.set_title(r'marginal $p(m_{' + str(index2) + '})$')  # Adding a title
    ax_marg_y.yaxis.tick_left()
    ax_marg_y.yaxis.set_label_position('left')
    ax_marg_y.legend()


def plot_joint_marginal_prior_posterior_comparison(prior_mean, prior_cov, posterior_mean, posterior_cov, index1, index2,
                                                   file_name="2_model_parameters_joint_marginal_prior_posterior_comparison.png"):
    """
    Plot the joint and marginal distributions of two model parameters for both prior and posterior distributions.
    :param prior_mean: The prior mean vector of the model parameters, size = (M, 1)
    :param prior_cov: The prior covariance matrix of the model parameters, size = (M, M)
    :param posterior_mean: The posterior mean vector of the model parameters, size = (M, 1)
    :param posterior_cov: The posterior covariance matrix of the model parameters, size = (M, M)
    :param index1: The index of the first model parameter
    :param index2: The index of the second model parameter
    :param file_name: The file name to save the plot
    """
    # 1) Initialize the figure and gridspec
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(4, 8)

    # 2) Joint and Marginals for Prior Distribution
    ax_joint_prior = fig.add_subplot(gs[1:4, 0:3])
    ax_x_prior = fig.add_subplot(gs[0, 0:3])
    ax_y_prior = fig.add_subplot(gs[1:4, 3])
    plot_joint_and_marginals(ax_joint=ax_joint_prior, ax_marg_x=ax_x_prior, ax_marg_y=ax_y_prior, mean=prior_mean, cov=prior_cov,
                             index1=index1, index2=index2, type='Prior')

    # 3) Joint and Marginals for Posterior Distribution
    ax_joint_post = fig.add_subplot(gs[1:4, 4:7])
    ax_x_post = fig.add_subplot(gs[0, 4:7])
    ax_y_post = fig.add_subplot(gs[1:4, 7])
    plot_joint_and_marginals(ax_joint=ax_joint_post, ax_marg_x=ax_x_post, ax_marg_y=ax_y_post, mean=posterior_mean, cov=posterior_cov,
                             index1=index1, index2=index2, type='Posterior')

    # 4) Save and show
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
