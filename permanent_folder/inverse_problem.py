from direct_problem import ADIMethod
import numpy as np
import pandas as pd
import time
from concurrent import futures
from typing import Tuple
import os
from utils import tikhonov_regularization, calculate_jacobian
import h5py

def minimize_equation(T_measured: np.ndarray, T_estimated: np.ndarray) -> float:
    """
    Calculates the difference between the real temperature and simulated temperature.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    T_simulated (np.ndarray): Simulated temperature data.

    Returns:
    float: Sum of squared differences.
    """
    return np.sum((T_estimated - T_measured) ** 2)

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float, tuple]) -> Tuple[float, int, np.ndarray]:
    """
    Calculates the difference in the objective function when a parameter is perturbed.

    Parameters:
    args (tuple): Contains index, original q, real temperature, delta, E_q, and regularization parameter.

    Returns:
    Tuple[float, int, np.ndarray]: The derivative estimate, the index and the perturbed temperature.
    """
    angular_position, heat_flux_original, T_measured, delta, error, lambda_regul, shape = args

    # Perturb the q parameter at the given index
    heat_flux_modified = heat_flux_original.copy()
    dheat_flux = heat_flux_original[angular_position] * delta
    heat_flux_modified[angular_position] += dheat_flux

    angular_size, radial_size = shape

    # Simulate the temperature with the modified q
    T_estimated_pertubed = ADIMethod(heat_flux_modified, radial_size, angular_size)[:, -1]

    # Calculate the new objective function value with regularization
    error_pertubed = minimize_equation(T_measured, T_estimated_pertubed) + (lambda_regul * tikhonov_regularization(heat_flux_modified))

    # Estimate the derivative using finite differences
    derivative = (error_pertubed - error) / dheat_flux

    return derivative, angular_position, T_estimated_pertubed

def compute_differences(heat_flux: np.ndarray, T_measured: np.ndarray, lambda_regul: float,
    executor: futures.Executor, delta: float, shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the derivatives of the objective function with respect to q.

    Parameters:
    q (np.ndarray): Current parameter vector.
    T_real (np.ndarray): Real temperature data.
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    delta (float): Perturbation size.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Array of derivatives, simulated temperature, and perturbed temperatures.
    """
    # Simulate the current temperature with the current q
    angular_size, radial_size = shape
    T_estimated = ADIMethod(heat_flux, radial_size, angular_size)[:, -1]  # Simulated temperature

    # Calculate the current objective function value with regularization
    objective_function = minimize_equation(T_measured, T_estimated)
    tikhonov = tikhonov_regularization(heat_flux)
    error =  objective_function + (lambda_regul * tikhonov)

    args = [(angular_position, heat_flux, T_measured, delta, error, lambda_regul, shape) for angular_position in range(angular_size)]

    results = list(executor.map(calculate_difference, args))

    # Initialize an array to store the derivatives
    gradient = np.zeros(angular_size, dtype=np.float64)
    pertubed_temperatures_list = np.zeros((angular_size, len(T_estimated)), dtype=np.float64)

    # Populate the derivative array
    for derivative, angular_position, T_estimated_pertubed in results:
        gradient[angular_position] = derivative # Diferença das derivadas dos erros
        pertubed_temperatures_list[angular_position] = T_estimated_pertubed

    return gradient, T_estimated, objective_function, tikhonov, pertubed_temperatures_list

def optimize_parameters(T_measured: np.ndarray, heat_flux: np.ndarray, lambda_regul: float,
    executor: futures.Executor, deviation: float, step_size: float,
    shape: tuple, max_iterations: int) -> Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]:
    """
    Optimizes the parameter vector q to minimize the objective function.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    q_initial (np.ndarray): Initial parameter vector.
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.
    step_size (float): Step size for optimization.
    max_iterations (int): Maximum number of iterations.

    Returns:
    Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]: Optimized q, minimized objective value, Tikhonov value, Temperature simulated, and optimization results.
    """
    path = "C:/Users/marce/Desktop/TCC/results/"

    iterations = 0
    delta = 1e-8
    objective_function = 1e5
    filename = path + f"data_{lambda_regul:.0e}_{deviation}_{max_iterations:.0e}"

    if os.path.exists(filename):
        hf = h5py.File(filename, 'w')
    else:
        hf = h5py.File(filename, 'x')

    angular_size = shape[0]

    start_time = time.time()
    start_cpu_time = time.process_time()

    # Morozov's discrepancy principle threshold
    morozov = angular_size * (deviation ** 2)
    with hf:

        hf.create_dataset('T_measured', data=T_measured)

        hf.attrs['Lambda'] = lambda_regul
        hf.attrs['Deviation'] = deviation
        hf.attrs['Morozov'] = morozov

        while iterations <= max_iterations and step_size >= 0: #and value_eq_min >= morozov
            # Compute derivatives and simulated temperatures
            gradient, T_estimated, objective_function, tikhonov, pertubed_temperatures_list = compute_differences(heat_flux, T_measured, lambda_regul, executor, delta, shape)

            # Print progress
            if iterations % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f'Iteration {iterations}, Objective Function: {objective_function:.6f}, '
                    f'Time: {elapsed_time:.2f}s, step_size: {step_size}, Morozov: {morozov}')
                start_time = time.time()

            iter_group = hf.create_group(f'iteration_{iterations}')
            
            # Salve o array heat_flux como um dataset dentro do grupo
            iter_group.create_dataset('heat_flux', data=heat_flux)
            iter_group.create_dataset('T_estimated', data=T_estimated)
            
            # Salve os valores escalares como atributos do grupo
            iter_group.attrs['step_size'] = step_size
            iter_group.attrs['Objective_Function'] = objective_function
            iter_group.attrs['Tikhonov'] = tikhonov
            iter_group.attrs['Time_Spent'] = elapsed_time

            jacobian = calculate_jacobian(heat_flux, T_estimated, pertubed_temperatures_list, delta)

            descent_direction = - gradient

            num = (jacobian @ descent_direction).T @ (T_measured - T_estimated)
            den = (jacobian @ descent_direction).T @ (jacobian @ descent_direction)
            step_size = num / den

            heat_flux = heat_flux + (step_size * descent_direction)

            iterations += 1

        end_cpu_time = time.process_time()
        cpu_time_used = end_cpu_time - start_cpu_time
        hf.attrs['CPU_time'] = cpu_time_used

    return heat_flux, T_estimated

def run_optimization(T_measured, max_iterations, lambda_regul: float, executor: futures.Executor,
                     deviation: float, shape: tuple) -> Tuple[np.ndarray, float, float]:
    """
    Runs the optimization process.

    Parameters:
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.

    Returns:
    Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]: Optimized q, minimized objective value, Tikhonov value, Temperature simulated, and optimization results.
    """
    # Initialize q with ones multiplied by 1000
    heat_flux_initial = np.ones(shape[0], dtype=np.float64) * 1000.0

    # Define step size
    step_size = 500
    
    args = optimize_parameters(
        T_measured          = T_measured, 
        heat_flux       = heat_flux_initial, 
        lambda_regul    = lambda_regul,
        executor        = executor,
        deviation       = deviation,
        step_size       = step_size,
        shape           = shape, 
        max_iterations  = max_iterations
        )

    return args