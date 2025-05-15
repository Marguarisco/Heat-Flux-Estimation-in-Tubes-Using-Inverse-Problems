import pandas as pd
import numpy as np

T_real_df = pd.read_csv('Transient File/data/temperature/Temperature_Boundary_External_9_80_1200.csv')

deviation_values =  [0.1,0.5,1,5,10,50,100,250]

results = []

for deviation in deviation_values:
    mesh_size, num_sensors = T_real_df.shape[1], 20
    reduction_factor = int(mesh_size / num_sensors)
    T_real = T_real_df.iloc[:, ::reduction_factor]
    Ntheta = T_real.shape[1]
    Nt = T_real.shape[0] 

    # Load or generate random values
    filename = f'Transient File/data/random_values/random_values_{Nt}x{num_sensors}.npy'
    random_values = np.load(filename)

    # Add deviation to simulated external temperatures
    T_real += (deviation * random_values)
    T_real = T_real.to_numpy()

    df_temp = pd.DataFrame({"Deviation": deviation, 
                            "Position": np.arange(1, num_sensors +1),
                            "Temperature": T_real[-1]})

    results.append(df_temp)

final_df = pd.concat(results, ignore_index=True) 
# Save the simulated external temperatures
final_df.to_csv(f'Transient File/data/temperature pertubed/random_values_{Nt}x{num_sensors}.csv', index=False)