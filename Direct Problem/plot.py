import numpy as np
from graph import *

initial_path ='C:/Users/marce/Desktop/TCC/Direct Problem/Results/resultados_'

method = 'adi'
nr = 9
ntheta = 80
time = 1000000

dados = np.load(initial_path + f'{method}_completo_{nr}_{ntheta}_{time}.npz')

T = dados['temperaturas']
dt = dados['dt']


print("Arquivo carregado com sucesso!")
print(f"Forma do array de temperaturas: {T.shape}")

graph_temp_time(T, method)

#graph_comparison_analitic(T, dt, method)

#gif_comparison_analitic(T, method)

graph_solo(T, dt, method)

gif_solo(T, dt, method)

heat_map_2d(T, dt, method)

gif_heatmap(T, dt, method)