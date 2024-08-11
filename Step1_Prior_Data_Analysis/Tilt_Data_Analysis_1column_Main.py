import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, linregress
import numpy as np


def detrend_data(data_segment, tilt_column):
    """
    Remove linear trend from the tilt data.
    :param data_segment: DataFrame segment of the tilt data.
    :param tilt_column: Name of the tilt data column ('tiltx.last' or 'tilty.last')
    :return: Detrended data as a numpy array and the trend line.
    """
    # Get indices (time) and values
    x = data_segment.index.values
    y = data_segment[tilt_column].values
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Calculate detrended values and the trend line
    detrended = y - (slope * x + intercept)
    trend_line = slope * x + intercept
    return detrended, trend_line


def plot_tilt_data(df, tilt_column, file_name="tilt_data_comparison.png"):
    """
    Function to plot original and detrended tilt data on the same graph, with subplots.
    :param df: The DataFrame containing the tilt data
    :param tilt_column: The name of the tilt data column ('tiltx.last' or 'tilty.last')
    :param file_name: The name of the file to save the plot
    """
    # Clean column name for display in titles and legends
    display_column = tilt_column.replace('.last', '')
    display_column = display_column.capitalize()

    # Adjust Time to start from zero, and convert to seconds
    df['Time Adjusted'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

    # Detrend the data
    detrended_data, trend_line = detrend_data(df.reset_index(), tilt_column)

    # Create the plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot original data and trend line on the left subplot
    ax1.plot(df['Time Adjusted'], df[tilt_column], label=f'Original {display_column}', color='blue')
    ax1.plot(df['Time Adjusted'], trend_line, label='Trend Line', linestyle='--', color='green')
    ax1.set_title(f'Original {display_column} with Linear Trend')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Measurement [microradian]')
    ax1.legend()

    # Plot detrended data on the right subplot
    ax2.plot(df['Time Adjusted'], detrended_data, label=f'Detrended {display_column}', color='red')
    ax2.set_title(f'Detrended {display_column}')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Measurement [microradian]')
    ax2.legend()

    plt.tight_layout()

    # Save and show the plot
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_distribution(df, tilt_column, file_name="distribution_analysis.png"):
    """
    Function to analyze the distribution of detrended tilt data and fit it with a Gaussian model.
    :param df: The dataframe containing the tilt data.
    :param tilt_column: The name of the column containing tilt data.
    :param file_name: The name of the file to save the plot.
    """
    display_column = tilt_column.replace('.last', '')
    display_column = display_column.capitalize()

    # Compute the detrended data
    df['Time Adjusted'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
    detrended_data, _ = detrend_data(df.reset_index(), tilt_column)

    # Visualize the distribution and fit a Gaussian model
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot histogram on ax1
    sns.histplot(detrended_data, kde=False, color='blue', bins=30, label='Histogram', ax=ax1, alpha=0.5)
    ax1.set_xlabel('Deviations from Mean [microradian]')
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Fit a Gaussian to the data
    mu, std = norm.fit(detrended_data)
    x_min, x_max = ax1.get_xlim()
    x = np.linspace(x_min, x_max, 100)

    # Create second y-axis for Gaussian fit
    ax2 = ax1.twinx()
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, 'k--', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')
    ax2.set_ylabel('Probability Density', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add title and legend
    fig.suptitle(f'Detrended {display_column}: Distribution Analysis with Gaussian Fit')
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Save and show the plot
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print('Tilt data analysis started: Measurements of tilt data from a single station with one device.')
    # Load the data
    data_path = 'tilt_longer_tiltx.csv'
    data = pd.read_csv(data_path)
    data['Time'] = pd.to_datetime(data['Time'], unit='ms')
    print("Data loaded successfully! The total number of data points is: ", len(data))

    # Check which tilt is present in the data
    # Assuming 'tiltx.last' is present, or replace with 'tilty.last' if needed
    tilt_column = 'tiltx.last' if 'tiltx.last' in data.columns else 'tilty.last'
    print(f"Data contains tilt data in column: {tilt_column}")

    # Visualization 1: original data (with a linear trend line) and detrended data
    file_name1 = tilt_column + "_data_comparison.png"
    plot_tilt_data(df=data, tilt_column=tilt_column, file_name=file_name1)
    # Visualization 2: Gaussian fit to the distribution of detrended data
    file_name2 = tilt_column + "_distribution_analysis.png"
    analyze_distribution(df=data, tilt_column=tilt_column, file_name=file_name2)
