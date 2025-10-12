import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 1. CONFIGURAÇÃO ---
# Edite os valores abaixo para selecionar o arquivo de resultado correto.
path = "permanent_folder/output/"

deviations = [0.1, 0.5]
deviation = deviations[1]  # Mude o índice para 0 se quiser o desvio de 0.1
last_iter_num= 198
print(deviation, last_iter_num)

lambda_list = np.logspace(-10, -5, num=10)
lambda_regul = lambda_list[5] # Mude o índice para escolher outro lambda

radial_size = 9
angular_size = 80
max_simulation_time = 19000

T_not_pertubed = f"permanent_folder/data/direct_problem_{radial_size}_{angular_size}_{max_simulation_time:.0e}.npz"
T_medida = np.load(T_not_pertubed)['estimated_temperature'][-1][::4]


# O nome do arquivo será montado a partir dos parâmetros acima
# Exemplo: data_2e-07_0.5_2e+03.hdf5
nome_arquivo = f"data_{lambda_regul:.2e}_{deviation}_1.00e+03"
caminho_completo_do_h5 = path + nome_arquivo
# --- FIM DA CONFIGURAÇÃO ---

print(f"Tentando carregar o arquivo: {caminho_completo_do_h5}")

try:
    with h5py.File(caminho_completo_do_h5, 'r') as hf:
        # Carrega a Temperatura Medida
        if 'T_measured' in hf:
            T_medida_ruido = hf['T_measured'][:]
        else:
            print("Dataset 'T_measured' não encontrado no arquivo.")
            exit()

        # Carrega a Temperatura Estimada da última iteração
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        if not iter_keys:
            print("Nenhuma iteração encontrada no arquivo para extrair a T_estimated.")
            exit()

        
        last_iter_group = hf[f'iteration_{last_iter_num}']
        
        if 'T_estimated' in last_iter_group:
            T_estimada = last_iter_group['T_estimated'][:]
        else:
            print(f"Dataset 'T_estimated' não encontrado na última iteração ({last_iter_num}).")
            exit()
            
    # --- 2. PREPARAÇÃO DOS DADOS PARA O EIXO X ---
    num_sensores = len(T_medida)
    theta_radianos = np.linspace(-np.pi, np.pi, num_sensores, endpoint=False) # Ângulos em radianos
    theta_graus = np.linspace(-180, 180, num_sensores, endpoint=False)          # Ângulos em graus para o eixo X
    
    Theta_completo = np.linspace(-np.pi,np.pi, num_sensores, endpoint=False) * 180/np.pi

    

    def make_cyclic(theta_array, array):
        # Converte ângulos de [-180, 180] para [0, 360]
        theta_0_360 = np.mod(theta_array, 360)
        
        # Obtém os índices que ordenam os novos ângulos
        sort_indices = np.argsort(theta_0_360)
        
        # Ordena os ângulos e as temperaturas
        theta_sorted = theta_0_360[sort_indices]
        array_sorted = array[sort_indices]
        
        # Adiciona o ponto 360, repetindo o dado do ponto 0 para fechar o ciclo
        theta_cyclic = np.append(theta_sorted, 360)
        array_cyclic = np.append(array_sorted, array_sorted[0])
    
        return theta_cyclic, array_cyclic

    # --- 3. GERAÇÃO DO GRÁFICO (COM NOVO ESTILO) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()

    erro_relativo_percent = np.zeros_like(T_medida)
        # Usamos np.divide para evitar erros quando T_analitico for zero
    np.divide(
            np.abs(T_estimada - T_medida), 
            T_medida, 
            out=erro_relativo_percent, 
            where=T_medida!=0
        )
    erro_relativo_percent *= 100 # Converter para porcentagem

    Theta_final_completo, T_medida = make_cyclic(Theta_completo, T_medida)
    _, T_estimada = make_cyclic(Theta_completo, T_estimada)
    _, erro_relativo_percent = make_cyclic(Theta_completo, erro_relativo_percent)
    _, T_medida_ruido = make_cyclic(Theta_completo, T_medida_ruido)

    Max_erro_temp =  np.max(erro_relativo_percent)
    print(Max_erro_temp)

    # Plota a Temperatura Medida (azul claro, linha contínua, marcador 'o')
    ax1.plot(
        Theta_final_completo,
        T_medida,
        marker='',
        markersize=4,
        linestyle='-',
        color='black', # Cor alterada para azul claro
        label='Temperatura Medida'
    )

    ax1.plot(
        Theta_final_completo,
        T_medida_ruido,
        linewidth = 1,
        markersize=4,
        linestyle='-.',    # Linha alterada para contínua
        color='grey', # Cor alterada para azul escuro
        label='Temperatura Medida c/ Ruído'
    )

    # Plota a Temperatura Estimada (azul escuro, linha contínua, marcador 'o')
    ax1.plot(
        Theta_final_completo,
        T_estimada,
        marker='o',       # Marcador alterado para 'o'
        markersize=4,
        linestyle='-',    # Linha alterada para contínua
        color='red', # Cor alterada para azul escuro
        label='Temperatura Estimada'
    )

    
        

    ax2.plot(Theta_final_completo, erro_relativo_percent, 'k--', lw = 1,  label="Erro relativo")
    # --- Configurações do Gráfico (estilo da imagem) ---
    ax1.set_xlabel("Ângulo (°)", fontsize=12)
    ax1.set_ylabel("Temperatura (K)", fontsize=12)
    ax1.grid(True, linestyle=':')
    
    # Legenda no canto superior esquerdo, 2 colunas, sem borda
    ax1.legend(loc='upper left', ncol=2, frameon=False)
    
    # Ajusta os limites do eixo Y
    min_temp = min(np.min(T_medida), np.min(T_estimada))
    max_temp = max(np.max(T_medida), np.max(T_estimada))
    ax1.set_ylim(min_temp - 5, max_temp + 5) # Adiciona uma pequena margem

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', ncol=1)

    # Configura os ticks do eixo X
    ax1.set_xticks(np.arange(0, 361, 60))
    ax2.set_ylabel('Erro relativo (%)')
    ax2.set_ylim(0, 1)

    # Formata o eixo para adicionar o sufixo "%"
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=2)) 

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Erro: Arquivo HDF5 não encontrado em '{caminho_completo_do_h5}'")
    print("Verifique se os parâmetros na seção de configuração correspondem a um arquivo existente.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")