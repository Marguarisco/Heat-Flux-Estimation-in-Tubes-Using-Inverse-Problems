import numpy as np
from graph import *

initial_path ='direct_problem_folder/Results/resultados_'

method = 'adi'

dados_adi = np.load(initial_path + f'{method}_completo_25_3_6000.npz')

method = 'implicit'

dados_implicit = np.load(initial_path + f'{method}_completo_25_3_6000.npz')

method = 'explicit'

dados_explicit = np.load(initial_path + f'{method}_completo_25_3_6000.npz')

T_adi = dados_adi['temperaturas']
dt_adi = dados_adi['dt']

T_implicit = dados_implicit['temperaturas']
dt_implicit = dados_implicit['dt']

T_explicit = dados_explicit['temperaturas']
dt_explicit = dados_explicit['dt']


#graph_comparison_analitic_all(T_explicit, T_implicit, T_adi, dt_adi)
#gif_comparison_analitic_all(T_explicit, T_implicit, T_adi, dt_adi)

graph_comparison_error(T_explicit, T_implicit, T_adi)