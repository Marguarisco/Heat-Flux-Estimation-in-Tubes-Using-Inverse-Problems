import numpy as np
from numba import njit
from scipy.integrate import simpson as simps
from typing import Tuple
from Direct_problem import ADIMethod
import os

@njit
def tikhonov_regularization(q: np.ndarray, use_differences: bool = True) -> float:
    """
    Calculates the Tikhonov regularization term.

    Parameters:
    q (np.ndarray): Parameter vector.
    use_differences (bool): Whether to use differences of q for regularization.

    Returns:
    float: Regularization term.
    """
    if not use_differences:
        return np.sum(q ** 2)
    else:
        diffs = np.diff(q)
        return np.sum(diffs ** 2)
@njit    
def calculate_jacobian(q: np.ndarray, simu_temp: np.ndarray, pert_temps: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    """
    Calculates the Jacobian J = dT/dq using finite differences.

    Parameters:
    q (np.ndarray): Current parameter vector.
    simu_temp (np.ndarray): Real temperature data.
    temps (np.ndarray): Simulated temperatures.
    delta (float): Small perturbation for finite difference approximation.

    Returns:
    np.ndarray: Jacobian vector J = [dT/dq_1, dT/dq_2, ..., dT/dq_n].
    """
    
    q_params = len(q)
    m_params = len(simu_temp)
    J = np.zeros((m_params, q_params), dtype=np.float64)

    for j in range(q_params):
        dq = q[j] * delta
        
        J[:,j] = (pert_temps[j] - simu_temp) / dq #dT/dq

    return J

def load_or_generate_random_values(mesh_size: int, filename: str = None) -> np.ndarray:
    """
    Loads random values from a file or generates them if the file does not exist.

    Parameters:
    mesh_size (int): Size of the random array to generate.
    filename (str): Filename for saving/loading random values.

    Returns:
    np.ndarray: Array of random values.
    """
    if os.path.exists(filename):
        # If the file exists, load the saved array
        values = np.load(filename)
    else:
        # If the file does not exist, generate a new array and save it
        values = np.random.normal(0, 1, mesh_size)
        np.save(filename, values)
    return values

def initializer():
    """
    Initializer function to precompute ADIMethod for caching purposes.
    """
    # Dummy q to initialize and cache ADIMethod
    q_dummy = np.ones(20, dtype=np.float64) * 100.0
    ADIMethod(q_dummy)

def run_experiment(radial_size, angular_size, max_time_steps):
    path = 'Permanent File/data/'

    # Teste para somente espacialmente
    Theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False) 

    # Define the heat source as a quadratic function of Theta
    heat_flux = ((-2000.0) * (Theta / np.pi) ** 2) + 2000.0

    # Execute the ADI method
    estimated_temperature = ADIMethod(heat_flux, radial_size, angular_size, max_time_steps)

    np.savez_compressed(
        path + f'Direct_Problem_{radial_size}_{angular_size}_{max_time_steps}.npz',
        heat_flux = heat_flux,
        estimated_temperature = estimated_temperature
    )
