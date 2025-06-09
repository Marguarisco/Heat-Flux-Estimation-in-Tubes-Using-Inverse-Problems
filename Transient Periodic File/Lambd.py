from Inverse_problem_param import run_optimization, heat_flux_approximation
import matplotlib.pyplot as plt
import numpy as np
from concurrent import futures
import pandas as pd
import multiprocessing as mp
import os
from utils import run_experiment, load_or_generate_random_values
from Direct_problem import ADIMethod

deviations = [0.1]
lambd = 0

radial_size = 9
angular_size = 80
experiment_time = 1000 
num_sensors = 20
max_iterations = 1000000
dt_real = 1 # Seconds
N_list = [6, 8, 10, 12]

if __name__ == '__main__':
    filename = f"Transient Periodic File/data/temperature/Temperature_Boundary_External_{radial_size}_{angular_size}_{experiment_time}.csv"

    if os.path.exists(filename):
        # If the file exists, load the saved array
        values = pd.read_csv(filename)
        print(f"Loaded array from {filename} and shape {values.shape}")
    else:
        # If the file does not exist, generate a new array and save it
        run_experiment(radial_size, angular_size, experiment_time, dt_real)
        values = pd.read_csv(filename)
        print(f"Made a new simulation and saved to {filename} with shape {values.shape}")


    T_real_df = values

    mesh_size = T_real_df.shape[1]
    reduction_factor = int(mesh_size / num_sensors)
    T_real = T_real_df.iloc[:, ::reduction_factor]
    
    shape = (experiment_time, T_real.shape[1], radial_size)

    # Load or generate random values
    filename = f'Transient Periodic File/data/random_values/random_values_{experiment_time}x{T_real.shape[1]}.npy'
    random_values = load_or_generate_random_values(T_real.shape, filename)

    # Get the number of available CPU cores
    num_processes = mp.cpu_count()

    # Create a directory named 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Iterate over each deviation value
        for deviation in deviations:
            # Iterate over each lambda value
            for N in N_list:
                summary_results = []
                detailed_results = []
                optimization_history = []

                # Print the current lambda and deviation being executed
                print(f"Executing for Lambda: {lambd}, Deviation: {deviation}, N: {N}")

                # Run the optimization process
                parameters, val_minimize, val_tikhonov, temperature_simulated, optimization_results, cpu_time_used = run_optimization(
                    T_real = T_real, 
                    random_values = random_values,
                    max_iterations = max_iterations,
                    lambda_regul = lambd, 
                    executor = executor, 
                    deviation = deviation,
                    shape = shape,
                    N = N)

                summary_results.append({
                        'Lambda': lambd,
                        'Deviation': deviation,
                        'Val_Tikhonov': val_tikhonov,
                        'Val_Minimize': val_minimize,
                        'CPU_Time': cpu_time_used
                    })
                
                for position in range(len(parameters)):
                    detailed_results.append({
                        'Lambda': lambd,
                        'Deviation': deviation,
                        'Position': position + 1,  # Inicia em 1
                        'parameters': parameters[position]
                    })

                optimization_history.extend(optimization_results.to_dict('records'))


                df_general_results = pd.DataFrame(summary_results)
                df_general_results.to_csv(f"Transient Periodic File/output/transient_summary_results_{radial_size}_{angular_size}_{experiment_time}_{N}_{max_iterations}.csv", index=False)

                df_detailed_results = pd.DataFrame(detailed_results)
                df_detailed_results.to_csv(f"Transient Periodic File/output/transient_detailed_results_{radial_size}_{angular_size}_{experiment_time}_{N}_{max_iterations}.csv", index=False)

                df_optimization_history = pd.DataFrame(optimization_history)
                df_optimization_history.to_csv(f"Transient Periodic File/output/transient_optimization_history_{radial_size}_{angular_size}_{experiment_time}_{N}_{max_iterations}.csv", index=False)
                
                heat_flux_final = heat_flux_approximation(parameters, num_sensors, experiment_time)
                df_heat_flux = pd.DataFrame(heat_flux_final)
                df_heat_flux.to_csv(f"Transient Periodic File/output/transient_heat_flux_{radial_size}_{angular_size}_{experiment_time}_{N}_{max_iterations}.csv", index=False)

                df_temperature = pd.DataFrame(ADIMethod(heat_flux_final, radial_size, num_sensors, experiment_time)[-1])
                df_temperature.to_csv(f"Transient Periodic File/output/transient_temperature_{radial_size}_{angular_size}_{experiment_time}_{N}_{max_iterations}.csv", index=False)

                print(f"CSVs for deviation {deviation} and N {N} saved successfully!")

    print(f"Finished! All results saved to 'results' directory.")


    