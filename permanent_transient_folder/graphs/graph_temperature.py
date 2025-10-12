import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# --- Parâmetros (ajuste conforme necessário para corresponder à sua simulação) ---
radial_size = 9
angular_size = 80
num_sensors = 20
total_simulation_time = 19000
max_iterations = 1000
deviation = 0.5
# O valor de alpha_regul é retirado de `main.py`: alpha_list[5:6] com np.logspace(-10, -5, num=10)
alpha_regul = np.logspace(-10, -5, num=10)[5]

# --- Caminhos para os arquivos ---
# Certifique-se de que estes caminhos estejam corretos para a sua estrutura de pastas
path_data = 'permanent_transient_folder/data/'
path_output = 'permanent_transient_folder/output/'

# --- Nomes dos arquivos ---
direct_problem_filename = os.path.join(path_data, f"direct_problem_{radial_size}_{angular_size}_{total_simulation_time:.0e}.npz")
optimization_results_filename = os.path.join(path_output, f"data_{alpha_regul:.0e}_{deviation}_{max_iterations:.0e}")

with np.load(direct_problem_filename) as data:
    T_real_full = data['estimated_temperature']

# --- Carregar Temperatura Simulada (do resultado da otimização) ---
try:
    with h5py.File(optimization_results_filename, 'r') as hf:
        # Encontrar a última iteração para obter o resultado final
        iterations = [int(key.split('_')[1]) for key in hf.keys() if key.startswith('iteration_')]
        if iterations:
            last_iteration = max(iterations)
            T_simulada = hf[f'iteration_{last_iteration}']['T_estimated'][:]
            print(f"Dados de temperatura simulada carregados da iteração {last_iteration} de: {optimization_results_filename}")
        else:
            print("Nenhuma iteração encontrada no arquivo de resultados da otimização.")
except FileNotFoundError:
    print(f"Erro: Arquivo de resultados da otimização não encontrado em '{optimization_results_filename}'")

shift_real = angular_size // 2
shift_simulado = num_sensors // 2

# Usa np.roll para rotacionar a primeira metade do array para o final
# O deslocamento é negativo para mover da esquerda para a direita
T_real_full = np.roll(T_real_full, shift=-shift_real, axis=1)
T_simulada = np.roll(T_simulada, shift=-shift_simulado, axis=1)

# --- Preparar dados para plotagem ---
# Reduzir a resolução da temperatura real para corresponder à temperatura simulada
# (baseado no número de sensores usados na otimização)
reduction_factor = int(angular_size / num_sensors)
T_real_full[0,:] = np.full(len(T_real_full[0]), 300)
T_real_reduced = T_real_full[:, ::reduction_factor]
T_simulada[0,:] = np.full(len(T_simulada[0]), 300)


# Calcular a diferença de temperatura
T_diff = T_simulada - T_real_reduced

# Configurar eixos para os gráficos
tempo = np.arange(total_simulation_time)
posicao_angular_real = np.linspace(0, 360, angular_size, endpoint=False)
posicao_angular_simulada = np.linspace(0, 360, num_sensors, endpoint=False)


# --- Gerar Gráficos ---
fig = plt.figure(figsize=(14, 10))
axs = fig.add_gridspec(2, 2)


temp_min = np.min([np.min(T_real_full), np.min(T_simulada)])
temp_max = np.max([np.max(T_real_full), np.max(T_simulada)])

# Gráfico 1: Temperatura Real
ax1 = fig.add_subplot(axs[0, 0])
im1 = ax1.imshow(T_real_full, extent=[0, 360, total_simulation_time, 0], aspect='auto', cmap='jet', vmin=temp_min, vmax=temp_max)
ax1.set_title('Temperatura Real (Sem Ruído)')
ax1.set_xlabel('Ângulo (°)')
ax1.set_ylabel('Tempo (s)')


fig.colorbar(im1, ax=ax1, label='Temperatura (K)')

# Gráfico 2: Temperatura Simulada
ax2 = fig.add_subplot(axs[0, 1])
im2 = ax2.imshow(T_simulada, extent=[0, 360, total_simulation_time, 0], aspect='auto', cmap='jet', vmin=temp_min, vmax=temp_max)
ax2.set_title('Temperatura Simulada')
ax2.set_xlabel('Ângulo (°)')
ax2.set_ylabel('Tempo (s)')
fig.colorbar(im2, ax=ax2, label='Temperatura (K)')

# Gráfico 3: Diferença de Temperatura
ax3 = fig.add_subplot(axs[1, :])
im3 = ax3.imshow(T_diff, extent=[0, 360, total_simulation_time, 0], aspect='auto', cmap='bwr')
ax3.set_xlabel('Ângulo (°)')
ax3.set_ylabel('Tempo (s)')
fig.colorbar(im3, ax=ax3, label='Diferença de Temperatura (K)')

plt.tight_layout()
plt.show()