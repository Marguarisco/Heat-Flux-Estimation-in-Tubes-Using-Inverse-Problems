import numpy as np
from numba import njit
from scipy.integrate import simpson as simps
from typing import Tuple
from direct_problem import ADIMethod
import os

@njit
def tikhonov_regularization(parameters: np.ndarray, use_differences: bool = True) -> float:
    """
    Calculates the Tikhonov regularization term.

    Parameters:
    parameters (np.ndarray): Parameter vector.
    use_differences (bool): Whether to use differences of parameters for regularization.

    Returns:
    float: Regularization term.
    """
    if not use_differences:
        return np.sum(parameters ** 2)
    else:
        diffs = np.diff(parameters)
        return np.sum(diffs ** 2)
    
@njit
def calculate_jacobian(parameters: np.ndarray, T_estimated: np.ndarray, historical_temperatures: np.ndarray, delta: float) -> np.ndarray:
    """
    Calculates the Jacobian J = dT/dp using finite differences.

    Parameters:
    parameters (np.ndarray): Current parameter vector.
    simu_temp (np.ndarray): Real temperature data.
    temps (np.ndarray): Simulated temperatures.
    delta (float): Small perturbation for finite difference approximation.

    Returns:
    np.ndarray: Jacobian vector J = [dT/dp_1, dT/dp_2, ..., dT/dp_n].
    """
    parameters_len = historical_temperatures.shape[0]
    iteration  = T_estimated.shape[0]
    pertu_len = T_estimated.shape[1]

    
    J = np.zeros((iteration, pertu_len, parameters_len), dtype=np.float64)

    for i in range(iteration):
        for j in range(parameters_len):
            dp = parameters[j] * delta

            J[i, :, j] = (historical_temperatures[j, i] - T_estimated[i]) / dp #dT/dp
    
    return J

def calculate_step(step: int, J: np.ndarray, direcao_descida: np.ndarray, T_measured: np.ndarray, T_estimated: np.ndarray, experiment_time: int, lambda_regul: int, parameters: np.ndarray) -> float:
    """
    Calculates the step size for optimization.

    Parameters:
    step (float): Step size.
    J (np.ndarray): Jacobian matrix. shape: (Nt, Ns, Np) → Nt = time steps, Ns = sensores, Np = nº de parâmetros
    direcao_descida (np.ndarray): Array of derivatives. shape: (Np,)
    T_real (np.ndarray): Real temperature data. shape: (Nt, 20)
    T_simulated (np.ndarray): Simulated temperature data. shape: (Nt, 20)
    Nt (int): Number of time steps. 
    lambda_regul (float): Regularization parameter.
    parameters (np.ndarray): Parameter vector.

    Returns:
    float: Step size.
    """
    
    # Termo da integral
    ddt_dx = np.tensordot(J, direcao_descida, axes=([2], [0]))  # shape: (Nt, 20)
    diff_temp = T_measured - T_estimated  # shape: (Nt, 20)
    
    integral_term = np.sum((diff_temp - step * ddt_dx) * (-ddt_dx), axis=1)  # shape: (Nt,)

    integral_result = simps(integral_term, x=np.arange(experiment_time)) # shape: scalar

    # Termo de regularização
    diff_p = np.diff(parameters) # shape: (19,)
    diff_d = np.diff(direcao_descida, axis=0) # shape: (19,)

    regularization_term = experiment_time * lambda_regul * np.sum((diff_p + (step * diff_d)) * diff_d) # shape: scalar

    return integral_result + regularization_term

def load_or_generate_random_values(shape: Tuple[int, int], filename: str = None) -> np.ndarray:
    """
    Loads random values from a file or generates them if the file does not exist.

    Parameters:
    shape (Tuple[int, int]): Shape of the random array to generate.
    filename (str): Filename for saving/loading random values.

    Returns:
    np.ndarray: Array of random values.
    """
    if filename is None:
        # Generate a filename based on the shape
        filename = f'random_values_{shape[0]}x{shape[1]}.npy'

    if os.path.exists(filename):
        # If the file exists, load the saved array
        values = np.load(filename)
        print(f"Loaded array from {filename} with shape {values.shape}")
    else:
        # If the file does not exist, generate a new array and save it
        values = np.random.normal(0, 1, shape)
        np.save(filename, values)
        print(f"Generated and saved new array to {filename} with shape {values.shape}")
    return values

def initializer():
    """
    Initializer function to precompute ADIMethod for caching purposes.
    """
    # Dummy q to initialize and cache ADIMethod
    q_dummy = np.ones(20, dtype=np.float64) * 100.0
    ADIMethod(q_dummy)

def run_experiment(filename, radial_size, angular_size, total_simulation_time, dt):
    theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False) 

    # Define the heat source as a quadratic function of Theta
    heat_flux = ((-2000.0) * (theta / np.pi) ** 2) + 2000.0
    
    # Execute the ADI method
    estimated_temperature = ADIMethod(heat_flux, radial_size, angular_size, total_simulation_time, dt)

    np.savez_compressed(
        filename,
        heat_flux = heat_flux,
        estimated_temperature = estimated_temperature
    )