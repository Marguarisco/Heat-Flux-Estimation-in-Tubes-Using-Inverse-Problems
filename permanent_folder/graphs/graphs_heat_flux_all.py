import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. FUNÇÃO PARA O Q IDEAL (baseada no seu utils.py) ---
def calcular_q_ideal(angular_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o heat flux ideal teórico.
    Retorna o Q ideal e os ângulos em graus para o eixo X.
    """
    # Em utils.py, theta vai de -pi a pi.
    theta_radianos = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)
    # Para o gráfico, usamos o eixo de 0 a 360 graus.
    theta_graus = np.linspace(0, 360, angular_size, endpoint=False)
    
    # Esta é a função do seu arquivo utils.py
    q_ideal = ((-2000.0) * (theta_radianos / np.pi) ** 2) + 2000.0
    
    return q_ideal, theta_graus

# --- 2. CONFIGURAÇÃO ---
# Caminho para a pasta que contém seus arquivos .h5 do problema permanente
caminho_da_pasta_resultados = "permanent_folder/output/"
padrao_arquivos = "data_*" # Padrão para encontrar os arquivos

# --- 3. LÓGICA PRINCIPAL ---

# Dicionário para agrupar os nomes dos arquivos por 'deviation'
arquivos_agrupados = {}
todos_os_arquivos = glob.glob(os.path.join(caminho_da_pasta_resultados, padrao_arquivos))

# Agrupa os arquivos lendo o atributo 'Deviation' de dentro de cada um
for arquivo in todos_os_arquivos:
    try:
        with h5py.File(arquivo, 'r') as hf:
            if 'Deviation' in hf.attrs:
                deviation_val = hf.attrs['Deviation']
                if deviation_val not in arquivos_agrupados:
                    arquivos_agrupados[deviation_val] = []
                arquivos_agrupados[deviation_val].append(arquivo)
    except Exception as e:
        print(f"Aviso: Não foi possível ler o arquivo '{os.path.basename(arquivo)}'. Pulando. Erro: {e}")

if not arquivos_agrupados:
    print("Nenhum arquivo de resultado encontrado ou agrupado.")
else:
    print(f"Arquivos agrupados por desvio: { {k: len(v) for k, v in arquivos_agrupados.items()} }")

# Itera sobre cada grupo de desvio para criar uma figura
for deviation, lista_de_arquivos in arquivos_agrupados.items():
    
    num_graficos = len(lista_de_arquivos)
    if num_graficos == 0:
        continue

    # Define o layout da grade de subplots
    ncols = 3 # Defina quantas colunas de mini-gráficos você quer
    nrows = math.ceil(num_graficos / ncols)
    
    # Cria a figura e a grade de eixos (subplots)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f'Comparação Q Ideal vs. Q Calculado para Desvio = {deviation}', fontsize=18)
    axes = axes.flatten()

    # Função auxiliar para ler o lambda de um arquivo para poder ordenar a lista
    def ler_lambda_do_arquivo(nome_arquivo):
        try:
            with h5py.File(nome_arquivo, 'r') as hf:
                return hf.attrs.get('Lambda', float('inf'))
        except:
            return float('inf')
            
    lista_de_arquivos.sort(key=ler_lambda_do_arquivo)
    
    # Itera sobre cada arquivo e seu eixo correspondente
    for i, arquivo_h5 in enumerate(lista_de_arquivos):
        ax = axes[i]
        
        try:
            with h5py.File(arquivo_h5, 'r') as hf:
                lambda_val = hf.attrs.get('Lambda', 'N/A')
                
                iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
                if not iter_keys: continue

                last_iter_num= 1000
                # O 'heat_flux' neste projeto é um vetor 1D
                q_calculado = hf[f'iteration_{last_iter_num}/heat_flux'][:]

                # O 'num_sensors' é o tamanho do vetor q_calculado
                num_sensors = q_calculado.shape[0]
                q_ideal, theta_graus = calcular_q_ideal(num_sensors)
                
                # Plota no subplot específico
                ax.plot(theta_graus, q_ideal, 'o-', markersize=3, color='lightskyblue', label='Q Ideal')
                ax.plot(theta_graus, q_calculado, 'o-', markersize=3, color='darkblue', label='Q Calculado')
                
                ax.set_title(f'λ = {lambda_val:.1e}')
                ax.grid(True, linestyle=':')
                ax.legend()
                ax.set_ylabel("Q (Heat Flux)")
                
        except Exception as e:
            ax.set_title(f"Erro ao ler {os.path.basename(arquivo_h5)}")
            ax.text(0.5, 0.5, f"Erro: {e}", ha='center', va='center', wrap=True)

    # Esconde os eixos que não foram usados
    for j in range(num_graficos, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

print("\nProcesso de geração de gráficos concluído!")