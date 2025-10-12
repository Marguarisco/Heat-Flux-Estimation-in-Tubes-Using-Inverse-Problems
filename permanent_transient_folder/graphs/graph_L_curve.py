import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


folder_path = "permanent_transient_folder\output"

all_results = []

file_paths = glob.glob(os.path.join(folder_path, 'data_*'))

print(f"Encontrados {len(file_paths)} arquivos para processar...")

for file_path in file_paths:
    try:
        with h5py.File(file_path, 'r') as hf:
            # Pega os parâmetros globais do arquivo
            lambda_val = hf.attrs.get('Lambda', np.nan)
            deviation_val = hf.attrs.get('Deviation', np.nan)

            # Encontra a última iteração
            iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
            if not iter_keys:
                print(f"  - Aviso: Arquivo {os.path.basename(file_path)} não contém iterações. Pulando.")
                continue
            
            last_iter_num = max([int(k.split('_')[1]) for k in iter_keys])
            last_iter_group = 1000
            last_iter_group = hf[f'iteration_{last_iter_num}']
            
            # Extrai os atributos da última iteração
            residual_norm = last_iter_group.attrs.get('Objective_Function')
            solution_norm = last_iter_group.attrs.get('Tikhonov')
            
            # Guarda os resultados
            all_results.append({
                'lambda': lambda_val,
                'deviation': deviation_val,
                'residual_norm': residual_norm,
                'solution_norm': solution_norm
            })
    except Exception as e:
        print(f"  - Erro ao processar o arquivo {os.path.basename(file_path)}: {e}")

# --- 3. ORGANIZAÇÃO DOS DADOS ---
# Cria um DataFrame do Pandas com todos os resultados
df = pd.DataFrame(all_results)

print("\nDados extraídos com sucesso:")
print(df.head())

# Agrupa os dados por valor de desvio
grouped_by_deviation = df.groupby('deviation')


# --- 4. GERAÇÃO DOS GRÁFICOS ---
print(f"\nGerando gráficos para {len(grouped_by_deviation)} valores de desvio encontrados...")

for deviation, group_df in grouped_by_deviation:
    sorted_group = group_df.sort_values('lambda')
    
    # --- MUDANÇA PRINCIPAL AQUI ---
    # Experimente valores onde a altura é maior que a largura
    # Exemplo 1: Proporção invertida
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Exemplo 2: Ainda mais "esticado"
    # plt.figure(figsize=(6, 10)) 
    # --- FIM DA MUDANÇA ---
    
    # Plota os pontos e a linha tracejada
    ax.plot(
        sorted_group['residual_norm'].to_numpy(), 
        sorted_group['solution_norm'].to_numpy(), 
        marker='D',           # Marcador de diamante
        linestyle='--',      # Linha tracejada
        color='blue',
        markersize=5
    )
    
    # ... (o resto do seu código continua exatamente o mesmo) ...
    i = 0
    # Adiciona os rótulos de lambda em cada ponto
    for _, row in sorted_group.iterrows():
        if i not in [1,2]:
            ax.text(
            row['residual_norm'], 
            row['solution_norm'],
            r'$\alpha=$'+f"{row['lambda']:.1e}", 
            fontsize=9,
            ha='left',
            va='bottom'
        )
        i +=1
    
    

    # Configurações do gráfico
    ax.set_xlabel(r'Norma da solução $||Ax_{\alpha} - b||_2$', fontsize=14)
    ax.set_ylabel(r'Norma da solução $||x_{\alpha}||_2$', fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which="both", linestyle=':')

    plt.show()

print("\nProcesso concluído!")