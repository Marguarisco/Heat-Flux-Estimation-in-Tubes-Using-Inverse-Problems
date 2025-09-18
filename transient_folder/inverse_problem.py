from direct_problem import ADIMethod
import numpy as np
from scipy.integrate import simpson as simps
from scipy.optimize import root_scalar
import time
from concurrent import futures
from typing import Tuple
import os
from utils import tikhonov_regularization, calculate_jacobian, calculate_step
import h5py

def minimize_equation(T_measured: np.ndarray, T_estimated: np.ndarray) -> float:
    """
    Calculates the difference between the real temperature and simulated temperature.

    Parameters:
    T_measured (np.ndarray): Real temperature data.
    T_estimated (np.ndarray): Simulated temperature data.

    Returns:
    float: Sum of integrated squared differences.
    """

    diff = (T_estimated - T_measured) ** 2

    return (1/2) * np.sum(simps(diff, x=np.arange(T_measured.shape[0]), axis=0))

def calculate_difference(args: Tuple[int, np.ndarray, np.ndarray, float, float, float, tuple]) -> Tuple[float, int, np.ndarray]:
    """
    Calculates the difference in the objective function when a parameter is perturbed.

    Parameters:
    args (tuple): Contains index, original q, real temperature, delta, E_q, and regularization parameter.

    Returns:
    Tuple[float, int, np.ndarray]: The derivative estimate, the index and the perturbed temperature.
    """
    angular_position, parameters, T_measured, delta, error, alpha_regul, shape = args

    parameters_modified = parameters.copy()
    dp = parameters[angular_position] * delta
    parameters_modified[angular_position] += dp

    total_simulation_time, angular_size, radial_size = shape

    heat_flux_modified = heat_flux_approximation(parameters_modified, angular_size, total_simulation_time)  # Current heat flux approximation

    # Simulate the temperature with the modified q
    T_estimated_pertubed = ADIMethod(heat_flux_modified, radial_size, angular_size, total_simulation_time)

    # Calculate the new objective function value with regularization
    error_pertubed = minimize_equation(T_measured, T_estimated_pertubed) + (alpha_regul * tikhonov_regularization(parameters_modified))

    # Estimate the derivative using finite differences
    derivative = (error_pertubed - error) / dp

    return derivative, angular_position, T_estimated_pertubed

def compute_differences(parameters: np.ndarray, T_measured: np.ndarray, alpha_regul: float,
    executor: futures.Executor, delta: float, shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the derivatives of the objective function.

    Parameters:
    parameters (np.ndarray): Current parameter vector.
    T_measured (np.ndarray): Real temperature data.
    alpha_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    delta (float): Perturbation size.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Array of derivatives, simulated temperature, and perturbed temperatures.
    """
    # Simulate the current temperature with the current q
    total_simulation_time, angular_size, radial_size = shape

    heat_flux = heat_flux_approximation(parameters, angular_size, total_simulation_time)  # Current heat flux approximation

    T_estimated = ADIMethod(heat_flux, radial_size, angular_size, total_simulation_time)  # Simulated temperature

    # Calculate the current objective function value with regularization
    objective_function = minimize_equation(T_measured, T_estimated)
    tikhonov = tikhonov_regularization(parameters)
    error = objective_function + (alpha_regul * tikhonov)

    args = [(angular_position, parameters, T_measured, delta, error, alpha_regul, shape) for angular_position in range(len(parameters))]

    results = list(executor.map(calculate_difference, args))

    # Initialize an array to store the derivatives
    gradient = np.zeros_like(parameters, dtype=np.float64)
    pertubed_temperatures_list = np.zeros((len(parameters), len(T_estimated), len(T_estimated[0])), dtype=np.float64)

    # Populate the derivative array
    for derivative, angular_position, T_estimated_pertubed in results:
        gradient[angular_position] = derivative # Diferença das derivadas dos erros
        pertubed_temperatures_list[angular_position] = T_estimated_pertubed

    return gradient, T_estimated, objective_function, tikhonov, pertubed_temperatures_list

def optimize_parameters(T_measured: np.ndarray, parameters: np.ndarray, alpha_regul: float,
    executor: futures.Executor, deviation: float, step_size: float,
    shape: tuple, max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimizes the parameter vector q to minimize the objective function.

    Parameters:
    T_measured (np.ndarray): Real temperature data.
    parameters (np.ndarray): Initial parameter vector.
    alpha_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.
    step_size (float): Step size for optimization.
    max_iterations (int): Maximum number of iterations.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Optimized q, minimized objective value, Tikhonov value, Temperature simulated, and optimization results.
    """

    path = "transient_folder/output/"
    if not os.path.exists(path):
        os.makedirs(path)

    delta = 1e-8
    N = int((len(parameters) - 1) / 4)
    filename = path + f"data_{deviation}_{max_iterations:.1e}_{N}"

    if os.path.exists(filename):
        print(f"Arquivo '{filename}' encontrado. Tentando resumir a otimização.")
        with h5py.File(filename, 'a') as hf: # Abre em modo de leitura para verificar
            iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
            if iter_keys:
                iter_nums = sorted([int(k.split('_')[-1]) for k in iter_keys])

                if len(iter_nums) >= 2:
                        # Se houver 2 ou mais iterações, escolhe a penúltima por segurança.
                        resume_iter_num = iter_nums[-2]
                        print(f"Múltiplas iterações encontradas. Resumindo da penúltima ({resume_iter_num}) por segurança.")
                else:
                    # Se houver apenas uma, não há outra escolha a não ser usá-la.
                    resume_iter_num = iter_nums[0]
                    print(f"Apenas uma iteração ({resume_iter_num}) encontrada. Resumindo a partir dela.")
                for num in iter_nums:
                    if num > resume_iter_num:
                        print(f"Removendo grupo de iteração incompleto/posterior: iteration_{num}")
                        del hf[f'iteration_{num}']

                print(f"Carregando estado da iteração {resume_iter_num}. A otimização continuará a partir da {resume_iter_num + 1}.")
                
                last_group = hf[f'iteration_{resume_iter_num}']
                # Carrega o estado da última iteração
                parameters = last_group['parameters'][:]
                objective_function = last_group.attrs['Objective_Function']
                step_size = last_group.attrs['step_size']
                iterations = resume_iter_num + 1
                if max_iterations <= resume_iter_num:
                    max_iterations = resume_iter_num + 10000
            else:
                # O arquivo existe, mas está vazio ou corrompido. Melhor começar de novo.
                print("Arquivo existe, mas não contém dados de iteração. Um novo arquivo será criado.")
                iterations = 0
                objective_function = 1e15
    else:
        print(f"Arquivo '{filename}' não encontrado. Iniciando nova otimização.")
        iterations = 0
        objective_function = 1e15

    angular_size = shape[1]
    total_simulation_time = shape[0]

    start_time = time.time()
    start_cpu_time = time.process_time()

    # Morozov's discrepancy principle threshold
    morozov = (1/2) * len(parameters) * total_simulation_time * (deviation ** 2)

    with h5py.File(filename, 'a') as hf:
        # Se for uma nova execução (iteration == 0), salva os metadados iniciais.
        if iterations == 0:
            if 'T_measured' in hf: del hf['T_measured'] # Remove se existir para evitar erro
            hf.create_dataset('T_measured', data=T_measured)
            hf.attrs['Lambda'] = alpha_regul
            hf.attrs['Deviation'] = deviation
            hf.attrs['Morozov'] = morozov
            hf.attrs['N'] = N

        try:
            while iterations <= max_iterations and step_size > 0: #and value_eq_min >= morozov:            
                # Compute derivatives and simulated temperatures
                iter_group = hf.create_group(f'iteration_{iterations}')

                if iterations % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f'Iteration {iterations}, Objective Function: {objective_function:,.6f}, '
                        f'Time: {elapsed_time:.2f}s, step_size: {step_size}, Morozov: {morozov}')
                    start_time = time.time()
                    iter_group.attrs['Time_Spent'] = elapsed_time
                    

                gradient, T_estimated, objective_function, tikhonov, pertubed_temperatures_list = compute_differences(parameters, T_measured, alpha_regul, executor, delta, shape)


                # Salve o array heat_flux como um dataset dentro do grupo
                iter_group.create_dataset('parameters', data=parameters)
                
                # Salve os valores escalares como atributos do grupo
                iter_group.attrs['step_size'] = step_size
                iter_group.attrs['Objective_Function'] = objective_function
                iter_group.attrs['Tikhonov'] = tikhonov

                jacobian = calculate_jacobian(parameters, T_estimated, pertubed_temperatures_list, delta)

                descent_direction = - gradient

                step_size = root_scalar(calculate_step, args=(jacobian, descent_direction, T_measured, T_estimated, total_simulation_time, alpha_regul, parameters), method='newton', x0=step_size).root

                parameters = parameters + (step_size * descent_direction)

                iterations += 1

        except KeyboardInterrupt:
            print("\n\n🛑 Otimização interrompida pelo usuário.")
            print("Salvando o progresso final e fechando o arquivo com segurança.")
            print("Você pode resumir a otimização executando o script novamente.")

        end_cpu_time = time.process_time()
        cpu_time_used = end_cpu_time - start_cpu_time
        hf.attrs['CPU_time'] = cpu_time_used

    return parameters, T_estimated

def run_optimization(T_measured, max_iterations, alpha_regul: float, executor: futures.Executor,
                     deviation: float, shape: tuple, N: int = 6) -> Tuple[np.ndarray, float, float]:
    """
    Runs the optimization process.

    Parameters:
    alpha_regul (float): Regularization parameter.
    executor (futures.Executor): Executor for parallel computation.
    deviation (float): Deviation for Morozov's discrepancy principle.

    Returns:
    """
    parameters_number = (4 * N) + 1

    # Initialize q
    parameters = np.ones(parameters_number, dtype=np.float64) * 100.0

    # Define step size
    step_size = 1

    args = optimize_parameters(
        T_measured          = T_measured, 
        parameters      = parameters, 
        alpha_regul    = alpha_regul,
        executor        = executor,
        deviation       = deviation,
        step_size       = step_size,
        shape           = shape, 
        max_iterations  = max_iterations
        )

    return args


def heat_flux_approximation(parameters: np.ndarray, angular_size: int, experimental_time) -> np.ndarray:
    """
    Approximates the heat flux based on the parameters.

    Parameters:
    parameters (np.ndarray): Parameters.
    ntheta (int): Number of angles.

    Returns:
    np.ndarray: Approximated heat flux.
    """

    N = int((len(parameters) - 1) / 4)

    A = parameters[0]
    
    B = parameters[1 : N + 1]
    C = parameters[N + 1 : 2 * N + 1]

    E = parameters[2 * N + 1 : 3 * N + 1]
    F = parameters[3 * N + 1 : 4 * N + 1]

    theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)

    w = np.pi # Frequency of the periodic function

    heat_flux = np.zeros(angular_size, dtype=np.float64)

    internal_theta =  w * (theta + np.pi) / (2 * np.pi)

    sum_cos_theta = np.sum([B[i] * np.cos((i + 1) * internal_theta) for i in range(N)], axis=0)
    sum_sin_theta = np.sum([C[i] * np.sin((i + 1) * internal_theta) for i in range(N)], axis=0)

    heat_flux_theta =  sum_cos_theta + sum_sin_theta

    internal_time = w * np.arange(experimental_time) / experimental_time
    
    sum_cos_time = np.sum([E[i] * np.cos((i + 1) * internal_time) for i in range(N)], axis=0)
    sum_sin_time = np.sum([F[i] * np.sin((i + 1) * internal_time) for i in range(N)], axis=0)
    
    heat_flux_time = sum_cos_time + sum_sin_time

    heat_flux = A + np.outer(heat_flux_time, heat_flux_theta)

    return heat_flux