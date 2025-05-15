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
    Calculates the difference between the real temperature and simulated temperature using Simpson's rule.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    T_simulated (np.ndarray): Simulated temperature data.

    Returns:
    float: The minimized equation value.
    """

    diff = (T_simulated - T_real) ** 2

    return np.sum(diff, axis=1)

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float, tuple]) -> Tuple[float, int]:
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
    dq = q_original[:,index] * delta
    q_modified[:,index] += dq
    
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

    # Prepare arguments for parallel computation
    args = [(i, q, T_real, delta, E_q, lambda_regul, shape) for i in range(len(q))]

    # Compute the derivatives in parallel
    results = list(executor.map(calculate_difference, args))

    # Initialize an array to store the derivatives
    gradiente = np.zeros_like(q, dtype=np.float64)
    perturbed_temps = np.zeros((len(q[0]), len(T_q), len(T_q[0])), dtype=np.float64)

    # Populate the derivative array
    for diff, pos, temp in results:
        gradiente[:, pos] = diff # Diferença das derivadas dos erros
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

    while iterations <= max_iterations: #and step_size >= 0: #and value_eq_min >= morozov
        # Compute derivatives and simulated temperatures
        gradiente, T_simulated, perturbed_temperatures = compute_differences(q, T_real, lambda_regul, executor, delta, shape)
        value_eq_min = minimize_equation(T_real, T_simulated)

        direcao_descida = - gradiente 

        for i in range(len(q)):
            J = calculate_jacobian(q[i], T_simulated, perturbed_temperatures, delta)

            step_size_unity = root_scalar(calculate_step, args=(J, direcao_descida[i], T_real, T_simulated, experiment_time, lambda_regul, q[i]), method='newton', x0=step_size).root
            step_size[i] = step_size_unity

        # Update the parameter vector
        q = q + (step_size * direcao_descida)

        # Print progress at every iteration or if Morozov condition is met
        if iterations % 1 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iteration {iterations}, Objective Function: {value_eq_min:.6f}, '
                  f'Time: {elapsed_time:.2f}s, step_size: {step_size[0]}, Morozov: {morozov}')
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

def run_optimization(T_real, random_values, max_iterations, lambda_regul: float, executor: futures.Executor,
                     deviation: float = 0.1, shape = None, output_csv: str = "optimization_results_transient.csv") -> Tuple[np.ndarray, float, float]:
    """
    Runs the optimization process.

    Parameters:
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.
    output_csv (str): Filename for saving optimization results.

    Returns:
    Tuple[np.ndarray, float, float]: Optimized q, minimized objective value, and Tikhonov value.
    """

    start_cpu_time = time.process_time()

    experiment_time, Ntheta, _ = shape

    # Add deviation to the real temperature data
    T_real += (deviation * random_values)
    T_real = T_real.to_numpy()

    # Initialize q with ones multiplied by 1000
    q_initial = np.ones((43200, Ntheta), dtype=np.float64) * 1000.0
    # Define step_size
    step_size = np.ones(43200, dtype=np.float64) * 0.1

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
    
    optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results = args

    # Save optimization history to CSV
    optimization_results.to_csv(output_csv, index=False)

    print("Optimization completed successfully.")
    print(f"Optimized q: {optimized_q}")
    print(f"Minimized Objective Function: {minimized_value}")
    print(f"Tikhonov Regularization Value: {tikhonov_val}")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    end_cpu_time = time.process_time()
    cpu_time_used = end_cpu_time - start_cpu_time

    return optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results, cpu_time_used



if __name__ == '__main__':
    # Regularization parameter
    lambda_reg = 1e-14
    deviation = 0.1

    # Determine the number of available CPU cores
    num_processes = mp.cpu_count()

    radial_size = 9
    angular_size = 80
    experiment_time = 43200 # 12 hours in seconds
    num_sensors = 20
    max_iterations = 1000

    filename = f"Transient Periodic File/data/temperature/Temperature_Boundary_External_{radial_size}_{angular_size}_{experiment_time}.csv"
    T_real_df = pd.read_csv(filename)

    mesh_size = T_real_df.shape[1]
    reduction_factor = int(mesh_size / num_sensors)
    T_real = T_real_df.iloc[:, ::reduction_factor]
    
    shape = (experiment_time, T_real.shape[1], radial_size)

    # Load or generate random values
    filename = f'Transient Periodic File/data/random_values/random_values_{experiment_time}x{T_real.shape[1]}.npy'
    random_values = load_or_generate_random_values(T_real.shape, filename)

    summary_results = []
    detailed_results = []
    optimization_history = []

    # Get the number of available CPU cores
    num_processes = mp.cpu_count()

    # Create a directory named 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create a process pool with the initializer
    with futures.ProcessPoolExecutor(max_workers=num_processes, initializer=initializer) as executor:
        optimized_q, minimized_value, tikhonov_val, temperature_simulated, optimization_results, cpu_time_used = run_optimization(
                    T_real = T_real, 
                    random_values = random_values,
                    max_iterations = max_iterations,
                    lambda_regul = lambda_reg, 
                    executor = executor, 
                    deviation = deviation,
                    shape = shape)
        
    output_csv: str = "Transient Periodic File/optimization_results_permanent.csv"
    optimization_results.to_csv(output_csv, index=False)

    print("Final Results:")
    print(f"Optimized q: {optimized_q}")
    print(f"Minimized Objective Function: {minimized_value}")
    print(f"Tikhonov Regularization Value: {tikhonov_val}")