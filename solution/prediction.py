from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model


def prepare_test_df(test_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Rearrange test dataframe shape for prediction."""

    missing_cols = list(set(df.columns) - set(test_df.columns))

    if missing_cols:
        # Add missing columns to test dataframe
        test_df = pd.concat([test_df, pd.DataFrame(columns=missing_cols)])
        test_df[missing_cols] = 0

    # Remove extra columns from test dataframe
    missing_cols = list(set(test_df.columns) - set(df.columns))
    test_df = test_df.drop(missing_cols, axis=1)

    # Reorder test dateframe columns
    return test_df[df.columns]


def predict(autoencoder: Model, data: pd.DataFrame, plot=False) -> pd.Series:
    """Make prediction using trained autoencoder"""

    # Predict the outpuxt for the data
    predicted_data = autoencoder.predict(data)
    # Calculate the reconstruction error for each row
    reconstruction_error = np.mean(np.square(predicted_data - data), axis=1)

    if plot:
        plt.hist(reconstruction_error, bins=50)
        plt.xlabel("Reconstruction error")
        plt.ylabel("Number of examples")
        plt.show()

    return reconstruction_error


def plot_reconstruction_error(
    reconstruction_error: pd.Series, threshold: float, ylim: Optional[tuple] = (0, 1)
):
    """Plot threshold and reconstruction error points"""
    plt.plot(
        reconstruction_error[reconstruction_error <= threshold],
        "bo",
        markersize=3,
        label="Normal",
    )
    plt.plot(
        reconstruction_error[reconstruction_error > threshold],
        "ro",
        markersize=3,
        label="Anomalous",
    )
    plt.axhline(y=threshold, color="g", linestyle="--", label="Threshold")
    plt.ylim(ylim)  # set y-axis limits
    plt.xlabel("Sample index")
    plt.ylabel("Reconstruction error")
    plt.title("Reconstruction error for normal and anomalous samples")
    plt.legend()
    plt.show()


def get_threshold(reconstruction_error: pd.Series, q: int) -> np.float64:
    """Calculate threshold value for reconstruction error"""
    return np.percentile(reconstruction_error, q)
