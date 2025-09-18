import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO ---
path = "permanent_folder/output/"

deviations = [0.1, 0.5]
deviation = deviations[1]
lambda_list = np.logspace(-10, -5, num=10)
lambda_regul = lambda_list[6]
# 2e-0(6) e 0.5(1)

print(f"{lambda_regul:.0e}")
# Cole o caminho completo para o arquivo HDF5 que contém o heat_flux a ser comparado
caminho_complepleto_do_h5 = path + f"data_{lambda_regul:.0e}_{deviation}_2e+03" # EX: ajuste para o seu arquivo
# --- FIM DA CONFIGURAÇÃO ---

try:
    with h5py.File(caminho_complepleto_do_h5, 'r') as hf:
        # Encontra a última iteração para pegar o heat_flux estimado
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        if not iter_keys:
            print("Nenhuma iteração encontrada no arquivo para extrair o heat_flux estimado.")
            exit() # Sai do script se não houver dados
            
        last_iter_num= 1000
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
    
    # Função ideal do heat_flux que você forneceu
    # ATENÇÃO: Se a sua função espera theta em GRAUS, mude 'theta_radianos' para 'theta_graus'
    q_ideal = ((-2000.0) * (theta_radianos / np.pi) ** 2) + 2000.0

    # --- 3. GERAÇÃO DO GRÁFICO ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plota o heat_flux ideal (azul mais claro, pontos conectados)
    ax.plot(
        theta_graus,       # Eixo X em graus
        q_ideal,
        marker='o',        # Marcador de círculo
        markersize=4,
        linestyle='-',     # Linha contínua
        color='lightskyblue', # Cor mais clara
        label='Sum of Q Ideal'
    )

    # Plota o heat_flux estimado (azul mais escuro, pontos conectados)
    ax.plot(
        theta_graus,      # Eixo X em graus
        q_estimado,
        marker='o',       # Marcador de círculo
        markersize=4,
        linestyle='-',    # Linha contínua
        color='darkblue', # Cor mais escura
        label='Sum of q'
    )
    
    # --- Adiciona os rótulos aos pontos ---
    # Rótulos para o Q Ideal (ajuste a frequência se tiver muitos pontos)
    # Exemplo: a cada 20 pontos
    for i in range(0, num_pontos, 20): # Ajuste o '20' para controlar quantos rótulos aparecem
        ax.text(
            theta_graus[i], 
            q_ideal[i], 
            f"{q_ideal[i]:.0f}", # Formata como inteiro, como na imagem
            fontsize=8, ha='center', va='bottom', color='lightskyblue'
        )
    
    # Rótulos para o Q Estimado (ajuste a frequência e a posição para não sobrepor)
    for i in range(0, num_pontos, 20): # Ajuste o '20'
        ax.text(
            theta_graus[i], 
            q_estimado[i], 
            f"{q_estimado[i]:.0f}", # Formata como inteiro
            fontsize=8, ha='center', va='top', color='darkblue'
        )
        
    # --- Configurações do Gráfico ---
    ax.set_xlabel("Ângulo", fontsize=12)
    ax.set_ylabel("Valor de Q", fontsize=12)
    ax.set_title("Comparação: Heat Flux Estimado vs. Ideal", fontsize=14)
    ax.grid(True, linestyle=':')
    ax.legend(loc='upper left', ncol=2, frameon=False) # Legenda no canto superior esquerdo, sem moldura
    
    # Ajusta os limites do eixo Y para que comece em zero ou um pouco abaixo
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(min(0, current_ymin), current_ymax * 1.05) # Começa em 0 ou abaixo, e com 5% de margem no topo

    # Configura os ticks do eixo X para mostrar de 0 a 360
    ax.set_xticks(np.arange(-180, 180, 50)) # Ticks a cada 50 graus
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Erro: Arquivo HDF5 não encontrado em '{caminho_complepleto_do_h5}'")
except Exception as e:
    print(f"Ocorreu um erro: {e}")