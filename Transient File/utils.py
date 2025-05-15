import numpy as np
from numba import njit
from scipy.integrate import simpson as simps
from typing import Tuple

from Direct_problem_periodic import ADIMethod
#from Direct_problem import ADIMethod

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
def calculate_jacobian(parameters: np.ndarray, simu_temp: np.ndarray, pert_temps: np.ndarray, delta: float) -> np.ndarray:
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
    parameters_len = pert_temps.shape[0]
    iteration  = simu_temp.shape[0]
    pertu_len = simu_temp.shape[1]

    
    J = np.zeros((iteration, pertu_len, parameters_len), dtype=np.float64)

    for i in range(iteration):
        for j in range(parameters_len):
            dp = parameters[j] * delta

            J[i, :, j] = (pert_temps[j, i] - simu_temp[i]) / dp #dT/dp
    
    return J

def calculate_step(step: int, J: np.ndarray, direcao_descida: np.ndarray, T_real: np.ndarray, T_simulated: np.ndarray, experiment_time: int, lambda_regul: int, parameters: np.ndarray) -> float:
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
    diff_temp = T_real - T_simulated  # shape: (Nt, 20)
    
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

def run_experiment(radial_size, angular_size, experimental_time):
    import pandas as pd

    '''
    # Define the theta distribution
    Theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False) 

    # Define the heat source as a quadratic function of Theta
    heat_flux = ((-2000.0) * (Theta / np.pi) ** 2) + 2000.0'''
    
    heat_flux = np.zeros((experimental_time), dtype=np.float64)

    for i in range(experimental_time):
        heat_flux[i] = 2000 * (1 + np.sin(np.pi * i / experimental_time))
    

    heat_flux_data = pd.DataFrame(heat_flux)
    heat_flux_data.to_csv(f'Transient File/data/heat_flux_real_{experimental_time}.csv', index=False)

    # Execute the ADI method
    T_ext_history = ADIMethod(heat_flux, radial_size, angular_size, experimental_time)

    # Create a DataFrame to store the results
    df = pd.DataFrame(T_ext_history, columns=[f'Theta {i}' for i in range(angular_size)])
    csv_filename = f'Transient File/data/temperature/Temperature_Boundary_External_{radial_size}_{angular_size}_{experimental_time}.csv'

    # Save the results to the CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")