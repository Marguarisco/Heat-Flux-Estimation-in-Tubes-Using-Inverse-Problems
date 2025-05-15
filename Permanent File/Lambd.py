from Problema_Inverso_Permanente import run_optimization
import matplotlib.pyplot as plt
import numpy as np
from concurrent import futures
import pandas as pd
import multiprocessing as mp
import os

# Define the values of lambda and deviation
deviations = [0.1, 0.5]
lambdas = np.logspace(-14, 0, num=15) 

summary_results = []
detailed_results = []
optimization_history = []

if __name__ == '__main__':
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