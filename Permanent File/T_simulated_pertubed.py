import pandas as pd
import numpy as np

T_real_df = pd.read_csv('Permanent File/T_simulated_9_80.csv')

deviation_values =  [0.1, 0.5]

results = []

for deviation in deviation_values:
    mesh_size, sensors = len(T_real_df), 20
    reduction_factor = int(mesh_size / sensors)
    T_real = T_real_df.iloc[:, -1].tolist()[::reduction_factor]
    mesh_size = len(T_real)

    # Load or generate random values
    filename = f'Permanent File/random_values_{mesh_size}.npy'
    random_values = np.load(filename)

    # Add deviation to simulated external temperatures
    T_pertubado = np.array(T_real) + (deviation * random_values)

    df_temp = pd.DataFrame({"Deviation": deviation, 
                            "Position": np.arange(1, sensors +1),
                            "Temperature": T_pertubado})

    results.append(df_temp)

final_df = pd.concat(results, ignore_index=True) 
# Save the simulated external temperatures
final_df.to_csv('Permanent File/T_simulated_9_80_pertubed.csv', index=False)