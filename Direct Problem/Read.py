import numpy as np
from graph import *

initial_path ='C:/Users/marce/Desktop/TCC/Direct Problem/Results/resultados_'

method = 'adi'

dados = np.load(initial_path + f'{method}_completo_25_3_6000.npz')

T = dados['temperaturas']
dt = dados['dt']


print("Arquivo carregado com sucesso!")
print(f"Forma do array de temperaturas: {T.shape}")

graph_temp_time(T, method)

graph_comparison_analitic(T, dt, method)

gif_comparativa(T, method)

#heat_map_2d(T, dt, method)

#gif_heatmap(T, dt, method)