import numpy as np
import pandas as pd

def norms_vs_exact(T_num):
    T_exact = pd.read_csv('C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv', header = None)

    Nt = T_num.shape[0]
    l2 = np.zeros(Nt)
    linf = np.zeros(Nt)
    for n in range(Nt):
        diff = T_num[n, -1, :] - T_exact.iloc[:, n]
        l2[n] = np.sqrt(np.mean(diff**2))
        linf[n] = np.max(np.abs(diff))
    return l2, linf

initial_path ='C:/Users/marce/Desktop/TCC/Direct Problem/Results/resultados_'

method = 'ADI'

dados = np.load(initial_path + f'{method}_completo_25_3_6000.npz')


T = dados['temperaturas']
running_time = dados['running_time']
running_time_process = dados['running_time_process']
l2, linf = norms_vs_exact(T)

print('running_time:', running_time, 
      'running_time_process:',running_time_process, 
      'l2:', l2[-1], 
      'linf:', linf[-1], 
      max(T[-1,-1,:]), min(T[-1,-1,:]))