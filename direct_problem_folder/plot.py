import numpy as np
from graph import *
from erro_relativo import *

initial_path ='direct_problem_folder/Results/resultados_'

method = 'ADI'
nr = 25
ntheta = 3
time = 6000

dados = np.load(initial_path + f'{method}_completo_{nr}_{ntheta}_{time}.npz')

T = dados['temperaturas']
dt = dados['dt']
running_time = dados['running_time']
running_time_CPU = dados['running_time_process']

print(running_time)

print("Arquivo carregado com sucesso!")
print(f"Forma do array de temperaturas: {T.shape}")

#graph_temp_time(T, method)

#graph_comparison_analitic(T, dt, method)

#gif_comparison_analitic(T, method)

#graph_solo(T, dt, method)

#gif_solo(T, dt, method)

#heat_map_2d(T, dt, method)

#gif_heatmap(T, dt, method)

#calcular_max_erro_relativo(T, dt)

#calcular_erro_apenas_ultimo_tempo(T,dt)