import matplotlib.pyplot as plt


def visualize_2d_matrix(matrix, file_name=None):
    """
    Visualize a 2D matrix.
    :param matrix: The 2D matrix to visualize
    :param xlabel: The label for the x-axis (default 'Index j')
    :param ylabel: The label for the y-axis (default 'Index i')
    :param cmap: The colormap to use (default 'viridis')
    :param file_name: The file name to save the figure. If None, the figure will not be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='jet', aspect='auto')
    ax.set_xlabel(r"Index $i$", fontsize=12)
    ax.set_ylabel(r"Index $j$", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title('')  # Remove colorbar title
    # Save the show
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_posterior_model_covariance_matrix(posterior_model_covariance, file_name="posterior_model_covariance_matrix.png"):
    """
    Plot the posterior model covariance matrix.
    :param posterior_model_covariance: The posterior model covariance matrix
    :param file_name: The file name to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(posterior_model_covariance, cmap='coolwarm')
    ax.set_title('Posterior Model Covariance Matrix', fontsize=14)
    ax.set_xlabel(r'index $i$', fontsize=12)
    ax.set_ylabel(r'index $j$', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title('')  # Remove colorbar title
    # Save and show
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_resolution_matrix(RM, trace_RM, file_name='model_resolution_matrix.png'):
    """
    Plot the posterior model covariance matrix RM.
    :param RM: The posterior model covariance matrix; size = (M, M)
    :param trace_RM: The trace of the RM matrix
    :param file_name: The file name to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(RM, cmap='jet')
    ax.set_title(r'Model Resolution Matrix $R_{{ij}}$. The Trace is {:.2f}'.format(trace_RM), fontsize=14)
    ax.set_xlabel(r'index $i$', fontsize=12)
    ax.set_ylabel(r'index $j$', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title('')
    # Save and show
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_resolution_matrix():
    pass


