import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO ---
# Edite os valores abaixo para selecionar o arquivo de resultado correto.
path = "permanent_folder/output/"

deviations = [0.1, 0.5]
deviation = deviations[0]  # Mude o índice para 0 se quiser o desvio de 0.1

lambda_list = np.logspace(-10, -5, num=10)
lambda_regul = lambda_list[6] # Mude o índice para escolher outro lambda

# O nome do arquivo será montado a partir dos parâmetros acima
# Exemplo: data_2e-07_0.5_2e+03.hdf5
nome_arquivo = f"data_{lambda_regul:.0e}_{deviation}_2e+03"
caminho_completo_do_h5 = path + nome_arquivo
# --- FIM DA CONFIGURAÇÃO ---

print(f"Tentando carregar o arquivo: {caminho_completo_do_h5}")

try:
    with h5py.File(caminho_completo_do_h5, 'r') as hf:
        # Carrega a Temperatura Medida
        if 'T_measured' in hf:
            T_medida = hf['T_measured'][:]
        else:
            print("Dataset 'T_measured' não encontrado no arquivo.")
            exit()

        # Carrega a Temperatura Estimada da última iteração
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        if not iter_keys:
            print("Nenhuma iteração encontrada no arquivo para extrair a T_estimated.")
            exit()
            
        last_iter_num= 1000
        last_iter_group = hf[f'iteration_{last_iter_num}']
        
        if 'T_estimated' in last_iter_group:
            T_estimada = last_iter_group['T_estimated'][:]
        else:
            print(f"Dataset 'T_estimated' não encontrado na última iteração ({last_iter_num}).")
            exit()
            
    # --- 2. PREPARAÇÃO DOS DADOS PARA O EIXO X ---
    num_sensores = len(T_medida)
    indices_sensores = np.arange(num_sensores)

    # --- 3. GERAÇÃO DO GRÁFICO (COM NOVO ESTILO) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plota a Temperatura Medida (azul claro, linha contínua, marcador 'o')
    ax.plot(
        indices_sensores,
        T_medida,
        marker='o',
        markersize=5,
        linestyle='-',
        color='lightskyblue', # Cor alterada para azul claro
        label='Temperatura Medida'
    )

    # Plota a Temperatura Estimada (azul escuro, linha contínua, marcador 'o')
    ax.plot(
        indices_sensores,
        T_estimada,
        marker='o',       # Marcador alterado para 'o'
        markersize=5,
        linestyle='-',    # Linha alterada para contínua
        color='darkblue', # Cor alterada para azul escuro
        label='Temperatura Estimada'
    )
        
    # --- Configurações do Gráfico (estilo da imagem) ---
    ax.set_xlabel("Índice do Sensor", fontsize=12)
    ax.set_ylabel("Temperatura (K)", fontsize=12)
    ax.set_title("Comparação: Temperatura Estimada vs. Medida", fontsize=14)
    ax.grid(True, linestyle=':')
    
    # Legenda no canto superior esquerdo, 2 colunas, sem borda
    ax.legend(loc='upper left', ncol=2, frameon=False)
    
    # Ajusta os limites do eixo Y
    min_temp = min(np.min(T_medida), np.min(T_estimada))
    max_temp = max(np.max(T_medida), np.max(T_estimada))
    ax.set_ylim(min_temp - 5, max_temp + 5) # Adiciona uma pequena margem

    # Configura os ticks do eixo X
    ax.set_xticks(np.arange(0, num_sensores, step=max(1, num_sensores//10)))
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Erro: Arquivo HDF5 não encontrado em '{caminho_completo_do_h5}'")
    print("Verifique se os parâmetros na seção de configuração correspondem a um arquivo existente.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")