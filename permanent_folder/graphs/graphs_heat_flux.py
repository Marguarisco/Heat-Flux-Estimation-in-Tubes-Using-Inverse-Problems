import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO ---
path = "permanent_folder/output/"

deviations = [0.1, 0.5]
deviation = deviations[1]
last_iter_num= 550
print(deviation, last_iter_num)

lambda_list = np.logspace(-10, -5, num=10)
lambda_regul = lambda_list[5]
# 2e-0(6) e 0.5(1)

print(f"{lambda_regul:.0e}")
# Cole o caminho completo para o arquivo HDF5 que contém o heat_flux a ser comparado
caminho_complepleto_do_h5 = path + f"data_{lambda_regul:.2e}_{deviation}_1.00e+03" # EX: ajuste para o seu arquivo
# --- FIM DA CONFIGURAÇÃO ---

try:
    with h5py.File(caminho_complepleto_do_h5, 'r') as hf:
        # Encontra a última iteração para pegar o heat_flux estimado
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        if not iter_keys:
            print("Nenhuma iteração encontrada no arquivo para extrair o heat_flux estimado.")
            exit() # Sai do script se não houver dados

        last_iter_group = hf[f'iteration_{last_iter_num}']
        
        # Carrega o heat_flux estimado
        if 'heat_flux' in last_iter_group:
            q_estimado = last_iter_group['heat_flux'][:]
        else:
            print(f"Dataset 'heat_flux' não encontrado na última iteração ({last_iter_num}).")
            exit() # Sai do script se não houver heat_flux
            
    # --- 2. GERAÇÃO DO HEAT_FLUX IDEAL ---
    # Assume que o domínio angular vai de 0 a 2*pi (360 graus)
    # e que tem o mesmo número de pontos que o q_estimado
    num_pontos = len(q_estimado)
    theta_radianos = np.linspace(-np.pi, np.pi, num_pontos, endpoint=False) # Ângulos em radianos
    theta_graus = np.linspace(-180, 180, num_pontos, endpoint=False)          # Ângulos em graus para o eixo X
    
    Theta_completo = np.linspace(-np.pi,np.pi, num_pontos, endpoint=False) * 180/np.pi

    

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
    
    q_ideal = ((-2000.0) * (theta_radianos / np.pi) ** 2) + 2000.0

    erro_relativo_percent = np.zeros_like(q_ideal)
    erro_relativo_percent = np.abs(q_estimado - q_ideal)

    Theta_final_completo, q_ideal = make_cyclic(Theta_completo, q_ideal)
    _, q_estimado = make_cyclic(Theta_completo, q_estimado)
    _, erro_relativo_percent = make_cyclic(Theta_completo, erro_relativo_percent)
    
    # --- 3. GERAÇÃO DO GRÁFICO ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()

    # Plota o heat_flux ideal (azul mais claro, pontos conectados)
    ax1.plot(
        Theta_final_completo,       # Eixo X em graus
        q_ideal,
        marker='',        # Marcador de círculo
        markersize=4,
        linestyle='-',     # Linha contínua
        color='black', # Cor mais clara
        label='Ideal'
    )

    # Plota o heat_flux estimado (azul mais escuro, pontos conectados)
    ax1.plot(
        Theta_final_completo,      # Eixo X em graus
        q_estimado,
        marker='o',       # Marcador de círculo
        markersize=4,
        linestyle='-',    # Linha contínua
        color='blue', # Cor mais escura
        label='Estimado'
    )
    
    ax2.plot(Theta_final_completo, erro_relativo_percent, 'k--', lw = 1,  label="Erro absoluto")
        
    # --- Configurações do Gráfico ---
    ax1.set_xlabel("Ângulo (°)", fontsize=12)
    ax1.set_ylabel(r'Fluxo de Calor (q) [$W/m^2$]', fontsize=12)
    ax1.grid(True, linestyle=':')
    ax1.legend(loc='upper left', ncol=2, frameon=False) # Legenda no canto superior esquerdo, sem moldura
    
    # Ajusta os limites do eixo Y para que comece em zero ou um pouco abaixo
    current_ymin, current_ymax = ax1.get_ylim()
    ax1.set_ylim(min(0, current_ymin), current_ymax * 1.15) # Começa em 0 ou abaixo, e com 5% de margem no topo

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', ncol=1)

    # Configura os ticks do eixo X para mostrar de 0 a 360
    ax1.set_xticks(np.arange(0, 361, 60))
    ax2.set_ylabel('Erro absoluto')
    ax2.set_ylim(0, 1000)
        # Formata o eixo para adicionar o sufixo "%"
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Erro: Arquivo HDF5 não encontrado em '{caminho_complepleto_do_h5}'")
except Exception as e:
    print(f"Ocorreu um erro: {e}")