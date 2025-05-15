import numpy as np
from Direct_problem import ADIMethod

# Parâmetros mock
num_r = 9
num_theta = 5
max_time_steps = 3  # pequeno para debug

# Gera um heat_flux com formato (max_time_steps, num_theta)
heat_flux = np.ones((max_time_steps, num_theta), dtype=np.float64) * 1000.0

# Chama o método com os dados mock
T_ext_history = ADIMethod(heat_flux, num_r, num_theta, max_time_steps)

# Exibe resultado só pra confirmar
print("Shape do resultado:", T_ext_history.shape)