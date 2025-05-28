from Direct_problem_periodic import ADIMethod
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
    float: Sum of integrated squared differences.
    """

    diff = (T_simulated - T_real) ** 2

    return np.sum(simps(diff, x=np.arange(T_real.shape[0]), axis=0))

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float, tuple]) -> Tuple[float, int]:
    """
    Calculates the difference in the objective function when a parameter is perturbed.

    Parameters:
    args (tuple): Contains index, original q, real temperature, delta, E_q, and regularization parameter.

    Returns:
    Tuple[float, int, np.ndarray]: The derivative estimate, the index and the perturbed temperature.
    """
    index, parameters_original, T_real, delta, E_p, lambda_regul, shape = args

    # Perturb the parameter
    parameters_modified = parameters_original.copy()
    dp = parameters_original[index] * delta
    parameters_modified[index] += dp
    
    max_time_steps, num_theta, num_r = shape

    q_modified = heat_flux_aproximation(parameters_modified,  max_time_steps)  # Current heat flux approximation

    # Simulate the temperature with the modified q
    T_p_delta = ADIMethod(q_modified, num_r, num_theta, max_time_steps)

    # Calculate the new objective function value with regularization
    E_p_delta = minimize_equation(T_real, T_p_delta) + (lambda_regul * tikhonov_regularization(parameters_modified))

    # Estimate the derivative using finite differences
    derivative = (E_p_delta - E_p) / dp

    return derivative, index, T_p_delta

def compute_differences(parameters: np.ndarray, T_real: np.ndarray, lambda_regul: float,
                       executor: futures.Executor, delta: float, shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the derivatives of the objective function with respect to q.

    Parameters:
    parameters (np.ndarray): Current parameter vector.
    T_real (np.ndarray): Real temperature data.
    lambda_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    delta (float): Perturbation size.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Array of derivatives, simulated temperature, and perturbed temperatures.
    """
    # Simulate the current temperature with the current q
    max_time_steps, num_theta, num_r = shape
    
    q = heat_flux_aproximation(parameters,  max_time_steps)  # Current heat flux approximation
    
    T_p = ADIMethod(q, num_r, num_theta, max_time_steps)  # Simulated temperature

    # Calculate the current objective function value with regularization
    E_p = minimize_equation(T_real, T_p) + (lambda_regul * tikhonov_regularization(parameters))

    # Prepare arguments for parallel computation
    args = [(i, parameters, T_real, delta, E_p, lambda_regul, shape) for i in range(len(parameters))]

    # Compute the derivatives in parallel
    results = list(executor.map(calculate_difference, args))

    # Initialize an array to store the derivatives
    gradiente = np.zeros(len(parameters), dtype=np.float64)
    perturbed_temps = np.zeros((len(parameters), len(T_p), len(T_p[0])), dtype=np.float64)

    # Populate the derivative array
    for diff, pos, temp in results:
        gradiente[pos] = diff # Diferença das derivadas dos erros
        perturbed_temps[pos] = temp

    return gradiente, T_p, perturbed_temps

def optimize_parameters(T_real: np.ndarray, parameters: np.ndarray, lambda_regul: float,
                       executor: futures.Executor, deviation: float, experiment_time: int, step_size: float,
                       shape: tuple, max_iterations: int) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Optimizes the parameter vector q to minimize the objective function.

    Parameters:
    T_real (np.ndarray): Real temperature data.
    parameters (np.ndarray): Initial parameter vector [A B C].
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
    num_theta = shape[1]

    optimization_results: List[Dict[str, float]] = []
    start_time = time.time()

    # Morozov's discrepancy principle threshold
    morozov = num_theta * experiment_time * (deviation ** 2) 

    while iterations <= max_iterations and step_size >= 0 and value_eq_min >= morozov:
        # Compute derivatives and simulated temperatures
        gradiente, T_simulated, perturbed_temperatures = compute_differences(parameters, T_real, lambda_regul, executor, delta, shape)
        value_eq_min = minimize_equation(T_real, T_simulated)

        J = calculate_jacobian(parameters, T_simulated, perturbed_temperatures, delta)

        direcao_descida = - gradiente 

        step_size = root_scalar(calculate_step, args=(J, direcao_descida, T_real, T_simulated, experiment_time, lambda_regul, parameters), method='newton', x0=step_size).root

        # Update the parameter vector
        parameters = parameters + (step_size * direcao_descida)

        # Print progress at every iteration or if Morozov condition is met
        if iterations % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iteration {iterations}, Objective Function: {value_eq_min:.6f}, '
                  f'Time: {elapsed_time:.2f}s, step_size: {step_size}, Morozov: {morozov}')
            start_time = time.time()

        # Calculate Tikhonov regularization
        tikhonov_val = tikhonov_regularization(parameters)

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
    final_tikhonov_value = tikhonov_regularization(parameters)

    # Save optimization history to DataFrame
    df_results = pd.DataFrame(optimization_results)

    return parameters, final_minimize_value, final_tikhonov_value, T_simulated, df_results

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

    # Initialize q parameters
    parameters = np.ones(2, dtype=np.float64) * 100.0

    # Define step_size
    step_size = 100

    # Run the optimization
    args = optimize_parameters(
        T_real          = T_real, 
        parameters      = parameters, 
        lambda_regul    = lambda_regul,
        executor        = executor,
        deviation       = deviation, 
        experiment_time = experiment_time, 
        step_size       = step_size,
        shape           = shape, 
        max_iterations  = max_iterations
        )
    
    parameters, minimized_value, tikhonov_val, temperature_simulated, optimization_results = args

    # Save optimization history to CSV
    optimization_results.to_csv(output_csv, index=False)

    print("Optimization completed successfully.")
    print(f"Parameters: {parameters}")
    print(f"Minimized Objective Function: {minimized_value}")
    print(f"Tikhonov Regularization Value: {tikhonov_val}")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    end_cpu_time = time.process_time()
    cpu_time_used = end_cpu_time - start_cpu_time

    return parameters, minimized_value, tikhonov_val, temperature_simulated, optimization_results, cpu_time_used

def heat_flux_aproximation(parameters: np.ndarray, experimental_time) -> np.ndarray:
    """
    Approximates the heat flux based on the parameters.

    Parameters:
    parameters (np.ndarray): Parameters [A, B].

    Returns:
    np.ndarray: Approximated heat flux.
    """

    A = parameters[0]
    B = parameters[1]

    heat_flux = np.zeros((experimental_time), dtype=np.float64)

    for i in range(experimental_time):
        heat_flux[i] = A *( 1 + np.sin((B * i)/experimental_time))


    '''N = 8

    A = parameters[0]
    B = parameters[1 : N + 1]
    C = parameters[N + 1 : 2 * N + 1]
    

    w = np.pi # Frequency of the periodic function

    heat_flux = np.zeros(experimental_time, dtype=np.float64)

    internal = w * np.arange(experimental_time) / experimental_time
    
    sum_cos = np.sum([B[i] * np.cos((i + 1) * internal) for i in range(N)], axis=0)
    sum_sin = np.sum([C[i] * np.sin((i + 1) * internal) for i in range(N)], axis=0)
    
    heat_flux = A + sum_cos + sum_sin'''

    return heat_flux

if __name__ == '__main__':
    # Regularization parameter
    lambda_reg = 0
    deviation = 0.1

    # Determine the number of available CPU cores
    num_processes = mp.cpu_count()

    radial_size = 9
    angular_size = 80
    experiment_time = 1100 # 12 hours in seconds
    num_sensors = 20
    max_iterations = 15000

    filename = f"Transient File/data/temperature/Temperature_Boundary_External_{radial_size}_{angular_size}_{experiment_time}.csv"
    T_real_df = pd.read_csv(filename)

    mesh_size = T_real_df.shape[1]
    reduction_factor = int(mesh_size / num_sensors)
    T_real = T_real_df.iloc[:, ::reduction_factor]
    
    shape = (experiment_time, T_real.shape[1], radial_size)

    # Load or generate random values
    filename = f'Transient File/data/random_values/random_values_{experiment_time}x{T_real.shape[1]}.npy'
    random_values = load_or_generate_random_values(T_real.shape, filename)

    summary_results = []
    detailed_results = []
    optimization_history = []

    # Get the number of available CPU cores
    num_processes = mp.cpu_count()

    # Create a directory named 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create a process pool with the initializer
    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        parameters, minimized_value, tikhonov_val, temperature_simulated, optimization_results, cpu_time_used = run_optimization(
                    T_real = T_real, 
                    random_values = random_values,
                    max_iterations = max_iterations,
                    lambda_regul = lambda_reg, 
                    executor = executor, 
                    deviation = deviation,
                    shape = shape)
        
    
    output_csv: str = "Transient File/optimization_results_permanent.csv"
    optimization_results.to_csv(output_csv, index=False)
    print("Final Results:")
    print(f"Parameters: {parameters}")
    print(f"Minimized Objective Function: {minimized_value}")
    print(f"Tikhonov Regularization Value: {tikhonov_val}")