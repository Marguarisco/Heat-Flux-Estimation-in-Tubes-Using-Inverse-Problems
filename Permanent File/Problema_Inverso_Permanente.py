from Problema_Direto_Permanente import ADIMethod
import numpy as np
import pandas as pd
import time
from concurrent import futures
import multiprocessing as mp
import os
from typing import Tuple, List
from numba import njit

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
    
    n_params = len(q)
    m_params = len(simu_temp)
    J = np.zeros((m_params, n_params), dtype=np.float64)

    for j in range(n_params):
        dq = q[j] * delta
        
        J[:,j] = (pert_temps[j] - simu_temp) / dq #dT/dq

    return J

def minimize_equation(T_real: np.ndarray, T_simulated: np.ndarray) -> float:
    """
    Calculates the difference between the real temperature and simulated temperature.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    T_simulated (np.ndarray): Simulated temperature data.

    Returns:
    float: Sum of squared differences.
    """
    return np.sum((T_simulated - T_real) ** 2)

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float]) -> Tuple[float, int, np.ndarray]:
    """
    Calculates the difference in the objective function when a parameter is perturbed.

    Parameters:
    args (tuple): Contains index, original q, real temperature, delta, E_q, and regularization parameter.

    Returns:
    Tuple[float, int, np.ndarray]: The derivative estimate, the index and the perturbed temperature.
    """
    index, q_original, T_real, delta, E_q, lambda_regul = args

    # Perturb the q parameter at the given index
    q_modified = q_original.copy()
    dq = q_original[index] * delta
    q_modified[index] += (dq)

    # Simulate the temperature with the modified q
    T_q_delta = ADIMethod(q_modified)[:, -1]

    # Calculate the new objective function value with regularization
    E_q_delta = minimize_equation(T_real, T_q_delta) + (lambda_regul * tikhonov_regularization(q_modified))

    # Estimate the derivative using finite differences
    derivative = (E_q_delta - E_q) / dq

    return derivative, index, T_q_delta

def compute_differences(q: np.ndarray, T_real: np.ndarray, lambda_regul: float,
    executor: futures.Executor, delta: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    T_q = ADIMethod(q)[:, -1]  # Simulated temperature

    # Calculate the current objective function value with regularization
    E_q = minimize_equation(T_real, T_q) + (lambda_regul * tikhonov_regularization(q))

    # Initialize an array to store the derivatives
    gradiente = np.zeros(len(q), dtype=np.float64)
    perturbed_temp = np.zeros((len(q), len(T_q)))
    
    # Prepare arguments for parallel computation
    args = [(i, q, T_real, delta, E_q, lambda_regul) for i in range(len(q))]

    # Compute the derivatives in parallel
    results = list(executor.map(calculate_difference, args))

    # Populate the derivative array
    for diff, pos, temp in results:
        gradiente[pos] = diff
        perturbed_temp[pos] = temp

    return gradiente, T_q, perturbed_temp

def optimize_parameters(T_real: np.ndarray, q_initial: np.ndarray, lambda_regul: float,
    executor: futures.Executor, deviation: float, step_size: float = 100.0,
    max_iterations: int = 1e3) -> Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]:
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
    iterations = 0
    q = q_initial.copy()
    value_eq_min = 1e3
    num_theta = len(q)
    
    optimization_results = []

    start_time = time.time()

    # Morozov's discrepancy principle threshold
    morozov = num_theta * (deviation ** 2)  

    while iterations <= max_iterations and step_size >= 0: #and value_eq_min >= morozov
        # Compute derivatives and simulated temperatures
        gradiente, T_simulated, perturbed_temperatures = compute_differences(q, T_real, lambda_regul, executor)
        value_eq_min = minimize_equation(T_real, T_simulated)

        direcao_descida = - gradiente

        J = calculate_jacobian(q, T_simulated, perturbed_temperatures)
        
        numerador = (J @ direcao_descida).T @ (T_real - T_simulated)
        denominador = (J @ direcao_descida).T @ (J @ direcao_descida)
        step_size = numerador / denominador

        q = q + (step_size * direcao_descida)

        # Print progress
        if iterations % 100 == 0 or value_eq_min < morozov:
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

    # Save optimization history to CSV
    df_results = pd.DataFrame(optimization_results)
    
    return q, final_minimize_value, final_tikhonov_value, T_simulated, df_results

def load_or_generate_random_values(mesh_size: int, filename: str = 'random_values.npy') -> np.ndarray:
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

def run_optimization(lambda_regul: float, executor: futures.Executor, 
    deviation: float = 0.1) -> Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]:
    """
    Runs the optimization process.

    Parameters:
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.

    Returns:
    Tuple[np.ndarray, float, float, np.ndarray, pd.DataFrame]: Optimized q, minimized objective value, Tikhonov value, Temperature simulated, and optimization results.
    """
    # Load real temperature data
    T_real_df = pd.read_csv('Permanent File/T_simulated_9_80.csv')

    mesh_size, sensors = len(T_real_df), 20
    reduction_factor = int(mesh_size / sensors)
    T_real = T_real_df.iloc[:, -1].tolist()[::reduction_factor]
    mesh_size = len(T_real)

    # Load or generate random values
    filename = f'Permanent File/random_values_{mesh_size}.npy'
    random_values = load_or_generate_random_values(mesh_size, filename)

    # Add deviation to simulated external temperatures
    T_real = np.array(T_real) + (deviation * random_values)  # Only T_ext

    # Initialize q with ones multiplied by 1000
    q_initial = np.ones(mesh_size, dtype=np.float64) * 1000.0
   
    # Set NumPy print options for better readability
    np.set_printoptions(suppress=True)

    # Define step size
    step_size = 500.0

    # Run the optimization
    optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results = optimize_parameters(
        T_real, q_initial, lambda_regul, executor, deviation,
        step_size, max_iterations = int(1000))

    return optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results

def initializer():
    """
    Initializer function to precompute ADIMethod for caching purposes.
    """
    # Dummy q to initialize and cache ADIMethod
    q_dummy = np.ones(20, dtype=np.float64) * 100.0
    ADIMethod(q_dummy)

if __name__ == '__main__':

    # Regularization parameter
    lambda_reg = 0.00001

    # Determine the number of available CPU cores
    num_processes = mp.cpu_count()

    # Create a process pool with the initializer
    with futures.ProcessPoolExecutor(max_workers=num_processes, initializer=initializer) as executor:
        # Run the optimization
        optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results = run_optimization(lambda_reg, executor)

    output_csv: str = "Permanent File/optimization_results_permanent.csv"
    optimization_results.to_csv(output_csv, index=False)

    print("Optimization completed successfully.")
    print(f"Optimized q: {optimized_q}")
    print(f"Minimized Objective Function: {minimized_value}")
    print(f"Tikhonov Regularization Value: {tikhonov_val}")
    print(f"Temperature Simulated: {temperature_simulated}")

