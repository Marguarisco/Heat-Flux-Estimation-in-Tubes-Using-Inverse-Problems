from inverse_problem import run_optimization
import numpy as np
from concurrent import futures
import multiprocessing as mp
import os
from utils import run_experiment, load_or_generate_random_values

path = 'transient_folder/data/'

deviations = [0.5]
alpha_regul = 0

radial_size = 9
angular_size = 80

num_sensors = 20
total_simulation_time = 3600
max_iterations = 200000

N_list = [3]

if __name__ == '__main__':
    filename = path + f"direct_problem_{radial_size}_{angular_size}_{total_simulation_time}.npz"

    if os.path.exists(filename):
        values = np.load(filename)['estimated_temperature']
        print(f"Loaded array from {filename} and shape {values.shape}")

    else:
        run_experiment(filename, radial_size, angular_size, total_simulation_time, dt = 0.01)
        values = np.load(filename)['estimated_temperature']
        print(f"Made a new simulation and saved to {filename} with shape {values.shape}")


    T_measured = values
    reduction_factor = int(angular_size / num_sensors)
    T_measured = T_measured[:, ::reduction_factor]
    
    shape = (total_simulation_time, num_sensors, radial_size)

    filename = path + f'random_values_{total_simulation_time}x{num_sensors}.npy'
    random_values = load_or_generate_random_values(T_measured.shape, filename)

    num_processes = mp.cpu_count()

    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        for deviation in deviations:
            T_measured += (deviation * random_values)
            for N in N_list:

                print(f"Executing for alpha: {alpha_regul:.0e}, Deviation: {deviation}, N: {N}")

                args = run_optimization(
                    T_measured = T_measured,
                    max_iterations = max_iterations,
                    alpha_regul = alpha_regul,
                    executor = executor, 
                    deviation = deviation,
                    shape = shape,
                    N = N)
                
                parameters, T_estimated = args

                print("Simulation Finished. Saving data.")