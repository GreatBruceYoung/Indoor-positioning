import numpy as np


def rmse_calculator(prediction, truth):
    """Calculate rmse"""
    rmse = np.sqrt(np.mean(np.sum((prediction- truth) ** 2, 1)))
    return rmse


def average_error_calculator(prediction, truth):
    """Calculate average error"""
    average_error = np.mean(np.sqrt(np.sum((prediction - truth) ** 2, 1)))
    return average_error