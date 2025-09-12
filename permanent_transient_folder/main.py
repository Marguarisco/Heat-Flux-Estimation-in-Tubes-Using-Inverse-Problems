from inverse_problem import run_optimization
import numpy as np
from concurrent import futures
import multiprocessing as mp
import os
from utils import run_experiment, load_or_generate_random_values

path = 'permanent_transient_folder/data/'

deviations = [0.1, 0.5]
lambda_list = np.logspace(-10, -5, num=10)
lambda_list = lambda_list[1:]

radial_size = 9
angular_size = 80

num_sensors = 20
total_simulation_time = 19000
max_iterations = 1000


if __name__ == '__main__':
    filename = path + f"direct_problem_{radial_size}_{angular_size}_{total_simulation_time:.0e}.npz"

    if os.path.exists(filename):
        # If the file exists, load the saved array
        values = np.load(filename)['estimated_temperature']
        print(f"Loaded array from {filename} and shape {values.shape}")

    else:
        # If the file does not exist, generate a new array and save it
        run_experiment(filename, radial_size, angular_size, total_simulation_time, dt = 0.01)
        values = np.load(filename)['estimated_temperature']
        print(f"Made a new simulation and saved to {filename} with shape {values.shape}")


    T_measured = values
    reduction_factor = int(angular_size / num_sensors)
    T_measured = T_measured[:, ::reduction_factor]
    
    shape = (total_simulation_time, num_sensors, radial_size)

    # Load or generate random values
    filename = path + f'random_values_{total_simulation_time}x{num_sensors}.npy'
    random_values = load_or_generate_random_values(T_measured.shape, filename)

    # Get the number of available CPU cores
    num_processes = mp.cpu_count()

    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Iterate over each deviation value
        for deviation in deviations:
            T_measured += (deviation * random_values)

            # Iterate over each lambda value
            for lambda_regul in lambda_list:
                # Print the current lambda and deviation being executed
                print(f"Executing for Lambda: {lambda_regul:.0e}, Deviation: {deviation}")

                # Run the optimization process
                args = run_optimization(
                    T_measured = T_measured,
                    max_iterations = max_iterations,
                    lambda_regul = lambda_regul,
                    executor = executor, 
                    deviation = deviation,
                    shape = shape)
                
                parameters, T_estimated = args

                print("Simulation Finished. Saving data.")