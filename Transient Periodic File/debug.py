'''from Problema_Inverso_Transiente import *

"""[
[ T(theta 0, iteracao 0), T(theta 1, iteracao 0), T(theta 2, iteracao 0)]
[ T(theta 0, iteracao 1), T(theta 1, iteracao 1), T(theta 2, iteracao 1)] 
[ T(theta 0, iteracao 2), T(theta 1, iteracao 2), T(theta 2, iteracao 2)]
]
-> Result: [[Sum(T-T)]]]"""

T_real = np.array([[1, 2, 3], 
                   [4, 5, 6],
                   [7, 8, 9],
                   [1, 1, 1]])

T_simulated = np.array([[1, 1, 1], 
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])

miniminze_equation_value = minimize_equation(T_real, T_simulated) #fica como [x, y, z] -> antes era [x]
q_sim = np.array([[1, 2, 3],    #iteração 0
                 [4, 5, 6],     #iteração 1
                 [7, 8, 9],
                 [10, 11 , 12]])    #iteração 2

q_new = q_sim.copy()
dq = q_sim[:, 0] * 2
q_new[:, 0] += dq

tikhonov = tikhonov_regularization(q_sim, use_differences=True) #fica como [x, y, z] -> antes era [x]

E_q_delta = miniminze_equation_value + (0.5 * tikhonov) #fica como [x, y, z] -> antes era [x]

derivative = (E_q_delta - [1, 2, 3, 4]) / dq #derivative fica como [x, y, z] -> antes era [x]

gradiente = np.zeros_like(q_new, dtype=float)

gradiente[:,1] = np.array([2,3,4,5])
'''



from Problema_Direto_Transiente import heat_flux_function
from utils import run_experiment
import numpy as np
Nr = 9
Ntheta = 80
N_max = 19000 
Theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False) 

# Define the theta distribution

heat_flux = ((-2000.0) * (Theta / np.pi) ** 2) + 4000.0


run_experiment(9, 80, 1900) #retorna o valor do fluxo de calor em um ponto específico