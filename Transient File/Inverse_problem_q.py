from Direct_problem import ADIMethod
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.optimize import root_scalar
import time
from concurrent import futures
import multiprocessing as mp
from typing import Tuple, List, Dict
from utils import *

def minimize_equation(T_real: np.ndarray, T_simulated: np.ndarray) -> float:
    """
    Calculates the difference between the real temperature and simulated temperature.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    T_simulated (np.ndarray): Simulated temperature data.

    Returns:
    float: Sum of integrated squared differences.
    """

    diff = (T_simulated - T_real) ** 2

    return np.sum(simps(diff, x=np.arange(T_real.shape[0]), axis=0))

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float, tuple]) -> Tuple[float, int, np.ndarray]:
    """
    Calculates the difference in the objective function when a parameter is perturbed.

    Parameters:
    args (tuple): Contains index, original q, real temperature, delta, E_q, and regularization parameter.

    Returns:
    Tuple[float, int, np.ndarray]: The derivative estimate, the index and the perturbed temperature.
    """
    index, q_original, T_real, delta, E_q, lambda_regul, shape = args

    # Perturb the q parameter at the given index
    q_modified = q_original.copy()
    dq = q_original[index] * delta
    q_modified[index] += dq

    max_time_steps, num_theta, num_r = shape

    # Simulate the temperature with the modified q
    T_q_delta = ADIMethod(q_modified, num_r, num_theta, max_time_steps)

    # Calculate the new objective function value with regularization
    E_q_delta = minimize_equation(T_real, T_q_delta) + (lambda_regul * tikhonov_regularization(q_modified))

    # Estimate the derivative using finite differences
    derivative = (E_q_delta - E_q) / dq

    return derivative, index, T_q_delta

def compute_differences(q: np.ndarray, T_real: np.ndarray, lambda_regul: float,
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
    max_time_steps, num_theta, num_r = shape
    
    T_q = ADIMethod(q, num_r, num_theta, max_time_steps)  # Simulated temperature

    # Calculate the current objective function value with regularization
    E_q = minimize_equation(T_real, T_q) + (lambda_regul * tikhonov_regularization(q))
    args = [(i, q, T_real, delta, E_q, lambda_regul, shape) for i in range(num_theta)]

    results = list(executor.map(calculate_difference, args))

    # Initialize an array to store the derivatives
    gradiente = np.zeros(num_theta, dtype=np.float64)
    perturbed_temps = np.zeros((num_theta, len(T_q), len(T_q[0])), dtype=np.float64)

    # Populate the derivative array
    for diff, pos, temp in results:
        gradiente[pos] = diff # Diferença das derivadas dos erros
        perturbed_temps[pos] = temp

    return gradiente, T_q, perturbed_temps

def optimize_parameters(T_real: np.ndarray, q_initial: np.ndarray, lambda_regul: float,
    executor: futures.Executor, deviation: float, experiment_time: int, step_size: float,
    shape: tuple, max_iterations: int) -> Tuple[np.ndarray, float, float, np.ndarray]:
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
    Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]: Optimized q, minimized objective value, Tikhonov value, Temperature simulated, and optimization results.
    """
    iterations = 0
    delta = 1e-8
    value_eq_min = 1e5

    q = q_initial.copy()
    num_theta = shape[1]

    optimization_results: List[Dict[str, float]] = []
    start_time = time.time()

    # Morozov's discrepancy principle threshold
    morozov = num_theta * experiment_time * (deviation ** 2)

    while iterations <= max_iterations and step_size >= 0: #and value_eq_min >= morozov
        # Compute derivatives and simulated temperatures
        gradiente, T_simulated, perturbed_temperatures = compute_differences(q, T_real, lambda_regul, executor, delta, shape)
        value_eq_min = minimize_equation(T_real, T_simulated)

        J = calculate_jacobian(q, T_simulated, perturbed_temperatures, delta)

        direcao_descida = - gradiente

        step_size = root_scalar(calculate_step, args=(J, direcao_descida, T_real, T_simulated, experiment_time, lambda_regul, q), method='newton', x0=step_size).root

        q = q + (step_size * direcao_descida)

        # Print progress
        if iterations % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iteration {iterations}, Objective Function: {value_eq_min:.6f}, '
                  f'Time: {elapsed_time:.2f}s, step_size: {step_size}, Morozov: {morozov}')
            start_time = time.time()

        # Calculate Tikhonov regularization
        tikhonov_val = tikhonov_regularization(q)

        # Record the results
        optimization_results.append({
            "Iteration": iterations,
            "Lambda": lambda_regul,
            "Deviation": deviation,
            "step_size": step_size,
            "Morozov": morozov,
            "Objective_Function": value_eq_min,
            "Tikhonov": tikhonov_val,
            "Time_Spent": time.time() - start_time
        })

        iterations += 1


    # Final objective and regularization values
    final_minimize_value = minimize_equation(T_real, T_simulated)
    final_tikhonov_value = tikhonov_regularization(q)

    # Save optimization history to DataFrame
    df_results = pd.DataFrame(optimization_results)

    return q, final_minimize_value, final_tikhonov_value, T_simulated, df_results

def run_optimization(T_real, max_iterations, lambda_regul: float, executor: futures.Executor,
                     deviation: float, shape: tuple) -> Tuple[np.ndarray, float, float]:
    """
    Runs the optimization process.

    Parameters:
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.

    Returns:
    Tuple[np.ndarray, float, float]: Optimized q, minimized objective value, and Tikhonov value.
    """

    start_cpu_time = time.process_time()

    # Initialize q with ones multiplied by 1000
    q_initial = np.ones(shape[1], dtype=np.float64) * 1000.0

    # Define step size
    step_size = 1

    # Run the optimization
    args = optimize_parameters(
        T_real          = T_real, 
        q_initial       = q_initial, 
        lambda_regul    = lambda_regul,
        executor        = executor,
        deviation       = deviation,
        experiment_time = experiment_time, 
        step_size       = step_size,
        shape           = shape, 
        max_iterations  = max_iterations
        )
    
    end_cpu_time = time.process_time()
    cpu_time_used = end_cpu_time - start_cpu_time

    return args, cpu_time_used