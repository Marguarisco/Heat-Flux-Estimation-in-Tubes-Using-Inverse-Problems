from Inverse_problem import run_optimization
import matplotlib.pyplot as plt
import numpy as np
from concurrent import futures
import pandas as pd
import multiprocessing as mp
import os
from utils import run_experiment, load_or_generate_random_values

path = 'Permanent File/data/'
# Define the values of lambda and deviation
deviations = [0.1, 0.5]
lambdas = np.logspace(-14, 0, num=15) 

radial_size = 9
angular_size = 80

num_sensors = 20
max_iterations = 3000


if __name__ == '__main__':
    filename = path + f"Direct_Problem_{radial_size}_{angular_size}_{max_iterations}.npz"

    if os.path.exists(filename):
        # If the file exists, load the saved array
        values = np.load(filename)['estimated_temperature']
        print(f"Loaded array from {filename} and shape {values.shape}")

    else:
        # If the file does not exist, generate a new array and save it
        run_experiment(radial_size, angular_size, max_iterations)
        values = np.load(filename)['estimated_temperature']
        print(f"Made a new simulation and saved to {filename} with shape {values.shape}")

    T_real = values

    reduction_factor = int(angular_size / num_sensors)
    T_real = T_real[::reduction_factor]
    
    shape = (num_sensors, radial_size)

    filename = path + f'random_values_{angular_size}.npy'
    random_values = load_or_generate_random_values(angular_size, filename)

    summary_results = []
    detailed_results = []
    optimization_history = []   

    # Get the number of available CPU cores
    num_processes = mp.cpu_count()

    # Create a directory named 'results' if it doesn't exist
    os.makedirs('results', exist_ok=True)

    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Iterate over each deviation value
        for deviation in deviations:
            # Iterate over each lambda value
            for lambd in lambdas:
                # Print the current lambda and deviation being executed
                print(f"Executing for Lambda: {lambd}, Deviation: {deviation}")

                # Run the optimization process
                q, val_minimize, val_tikhonov, temperature_simulated, optimization_results = run_optimization(
                    lambd, executor, deviation=deviation)
                

                summary_results.append({
                        'Lambda': lambd,
                        'Deviation': deviation,
                        'Val_Tikhonov': val_tikhonov,
                        'Val_Minimize': val_minimize
                    })
                
                for position in range(len(q)):
                    detailed_results.append({
                        'Lambda': lambd,
                        'Deviation': deviation,
                        'Position': position + 1,  # Inicia em 1
                        'q': q[position],
                        'T': temperature_simulated[position]
                    })

                optimization_history.extend(optimization_results.to_dict('records'))


    df_general_results = pd.DataFrame(summary_results)
    df_general_results.to_csv("results/permanent_summary_results.csv", index=False)

    df_detailed_results = pd.DataFrame(detailed_results)
    df_detailed_results.to_csv("results/permanent_detailed_results.csv", index=False)

    df_optimization_history = pd.DataFrame(optimization_history)
    df_optimization_history.to_csv("results/permanent_optimization_history.csv", index=False)
    
    print("CSVs saved successfully!")