import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. CONFIGURAÇÃO ---
# Caminho para a pasta que contém seus arquivos .h5
caminho_da_pasta_resultados = "permanent_folder/output/"
padrao_arquivos = "data_*" # Padrão para encontrar os arquivos de dados

# --- 2. LÓGICA PRINCIPAL ---
caminho_dados_limpos = "permanent_folder/data/direct_problem_9_80_1e+10.npz"
try:
    dados_npz = np.load(caminho_dados_limpos)
    # Pega a matriz completa de temperatura do arquivo .npz
    T_completo_limpo = dados_npz['estimated_temperature']
    # Seleciona apenas a temperatura na superfície externa (última coluna)
    T_superficie_limpo_original = T_completo_limpo[:, -1]
    dados_limpos_carregados = True
    print(f"Dados de temperatura 'limpa' carregados com sucesso de '{caminho_dados_limpos}'")
except FileNotFoundError:
    print(f"AVISO: Arquivo de dados limpos não encontrado em '{caminho_dados_limpos}'.")
    print("A linha de 'T Limpo (Ideal)' não será plotada.")
    dados_limpos_carregados = False

# Dicionário para agrupar os nomes dos arquivos por 'deviation'
arquivos_agrupados = {}
# Usa o glob para encontrar todos os arquivos que correspondem ao padrão
todos_os_arquivos = glob.glob(os.path.join(caminho_da_pasta_resultados, padrao_arquivos))

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

# Itera sobre cada grupo de desvio para criar uma figura separada
for deviation, lista_de_arquivos in arquivos_agrupados.items():
    
    num_graficos = len(lista_de_arquivos)
    if num_graficos == 0:
        continue

    # Define o layout da grade de subplots
    ncols = 3 
    nrows = math.ceil(num_graficos / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f'Comparação de Temperaturas para Desvio = {deviation}', fontsize=18)
    axes = axes.flatten()

    def ler_lambda_do_arquivo(nome_arquivo):
        try:
            with h5py.File(nome_arquivo, 'r') as hf:
                return hf.attrs.get('Lambda', float('inf'))
        except:
            return float('inf')
            
    lista_de_arquivos.sort(key=ler_lambda_do_arquivo)
    
    for i, arquivo_h5 in enumerate(lista_de_arquivos):
        ax = axes[i]
        
        try:
            with h5py.File(arquivo_h5, 'r') as hf:
                lambda_val = hf.attrs.get('Lambda', 'N/A')
                T_medida = hf['T_measured'][:]
                
                iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
                if not iter_keys: continue
                
                last_iter_num= 1000
                T_estimada = hf[f'iteration_{last_iter_num}/T_estimated'][:]

                indices_sensores = np.arange(len(T_medida))
                
                # --- Plot das temperaturas Medida e Estimada ---
                ax.plot(indices_sensores, T_medida, 'o-', markersize=3, color='lightskyblue', label='T Medida (com ruído)')
                ax.plot(indices_sensores, T_estimada, 'o-', markersize=3, color='darkblue', label='T Estimada (final)')
                
                # --- LÓGICA PARA ADICIONAR A TEMPERATURA LIMPA ---
                if dados_limpos_carregados:
                    # Calcula o fator de redução para compatibilizar os dados
                    num_sensores = len(T_medida)
                    angular_size_original = len(T_superficie_limpo_original)
                    reduction_factor = angular_size_original // num_sensores
                    
                    T_limpo_reduzido = T_superficie_limpo_original[::reduction_factor]
                    
                    # Plota a nova linha
                    ax.plot(indices_sensores, T_limpo_reduzido, 'o-', markersize=3, color='green', label='T Limpo (Ideal)')

                ax.set_title(f'λ = {lambda_val:.1e}')
                ax.grid(True, linestyle=':')
                ax.legend()
                ax.set_xlabel("Índice do Sensor")
                ax.set_ylabel("Temperatura (K)")
                
        except Exception as e:
            ax.set_title(f"Erro ao ler {os.path.basename(arquivo_h5)}")
            ax.text(0.5, 0.5, f"Erro: {e}", ha='center', va='center', wrap=True)

    # Esconde os eixos que não foram usados
    for j in range(num_graficos, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

print("\nProcesso de geração de gráficos concluído!")