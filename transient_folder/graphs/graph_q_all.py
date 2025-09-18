import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from inverse_problem import heat_flux_approximation, ADIMethod 

# --- 1. FUNÇÃO PARA GERAR O HEAT FLUX REAL (IDEAL) ---
def gerar_heat_flux_real(experimental_time: int, angular_size: int) -> np.ndarray:
    """
    Gera o campo de fluxo de calor real (ideal) ao longo do tempo.
    Baseado na função run_experiment do seu utils.py.
    """
    theta = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)
    
    # Perfil de calor base (no espaço)
    heat_flux_base = ((-2000.0) * (theta / np.pi) ** 2) + 4000.0
    
    # Aplica a variação senoidal no tempo
    heat_flux_real_no_tempo = np.zeros((experimental_time, angular_size))
    for i in range(experimental_time):
        heat_flux_real_no_tempo[i] = heat_flux_base * (1 + np.sin(np.pi * i / experimental_time))
        
    return heat_flux_real_no_tempo

# --- 2. CONFIGURAÇÃO ---
# Coloque o caminho para o arquivo .h5 que você quer analisar
# Este deve ser um dos seus melhores resultados
caminho_arquivo_h5 = "transient_folder/output/data_0.1_1.0e+04_1"

# --- 3. LÓGICA PRINCIPAL ---

with h5py.File(caminho_arquivo_h5, 'r') as hf:
    print(f"Analisando o arquivo: {os.path.basename(caminho_arquivo_h5)}")
    
    # --- a. Carrega os dados do arquivo H5 ---
    
    T_measured = hf['T_measured'][:]
    
    iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
    if not iter_keys:
        raise ValueError("Nenhuma iteração encontrada no arquivo.")
        
    last_iter_num = max([int(k.split('_')[1]) for k in iter_keys])
    last_iter_num = 100

    experimental_time, num_sensors = T_measured.shape
    
    # Carrega o heat_flux estimado e a temperatura estimada da última iteração
    parameters = hf[f'iteration_{last_iter_num-1}/parameters'][:]

    q_estimated = heat_flux_approximation(parameters, num_sensors, experimental_time)

    T_estimated = ADIMethod(q_estimated, 9, num_sensors, experimental_time)
    
    # --- b. Prepara os dados para os gráficos ---
    
    # Gera o heat_flux real com a mesma dimensão espacial do estimado
    q_real = gerar_heat_flux_real(experimental_time, num_sensors)
    
    # Calcula a diferença
    diferenca_q = q_real - q_estimated
    
    # Pega o perfil de temperatura no último instante de tempo
    T_real_final = T_measured[:]
    T_estimada_final = T_estimated[:]
    
    # Vetor de ângulos para o eixo X
    theta_axis_temp = np.linspace(0, 360, num_sensors, endpoint=False)
    theta_axis_q_real = np.linspace(0, 360, q_real.shape[1], endpoint=False)
    
    # --- c. Cria o painel de 4 gráficos (2x2) ---
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Análise de Resultados - {os.path.basename(caminho_arquivo_h5)}', fontsize=16)

    # --- Gráfico 1: Real Heat Flux ---
    ax1 = axs[0, 0]
    im1 = ax1.imshow(q_real, aspect='auto', origin='lower', extent=[0, q_real.shape[1], 0, experimental_time])
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Real Heat Flux')
    ax1.set_xlabel('Theta (índice)')
    ax1.set_ylabel('Tempo (índice)')

    # --- Gráfico 2: Estimated Heat Flux ---
    ax2 = axs[0, 1]
    im2 = ax2.imshow(q_estimated, aspect='auto', origin='lower', extent=[0, num_sensors, 0, experimental_time])
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('Estimated Heat Flux')
    ax2.set_xlabel('Theta (índice)')
    ax2.set_ylabel('Tempo (índice)')

    # --- Gráfico 3: Diferença ---
    ax3 = axs[1, 0]
    # Centraliza a barra de cores no zero para ver desvios positivos (vermelho) e negativos (azul)
    norm = TwoSlopeNorm(vmin=-5000, vcenter=0, vmax=7000)
    im3 = ax3.imshow(diferenca_q, aspect='auto', origin='lower', cmap='seismic', extent=[0, num_sensors, 0, experimental_time], norm = norm)
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('Diferença (Real - Estimado)')
    ax3.set_xlabel('Theta (índice)')
    ax3.set_ylabel('Tempo (índice)')

    # --- Gráfico 4: Perfil de Temperatura Final ---
    ax4 = axs[1, 1]
    ax4.plot(theta_axis_temp, T_real_final[-1], label='Temperatura Real')
    ax4.plot(theta_axis_temp, T_estimada_final[-1], label='Temperatura Estimada')
    ax4.set_title('Temperatura vs Theta (Tempo Final)')
    ax4.set_xlabel('Theta (Graus)')
    ax4.set_ylabel('Temperatura')
    ax4.set_ylim(bottom=300) # Define o limite inferior do eixo Y, como na imagem
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
