import h5py
import pandas as pd
import matplotlib.pyplot as plt

caminho_completo  = "permanent_transient_folder/output/data_6e-08_0.5_1e+03"
lambda_problematico = 8e-07

try:
    with h5py.File(caminho_completo, 'r') as hf:
        dados_iteracao = []
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        
        num_iteracoes = len(iter_keys)
        max_iter_salvo = hf.attrs.get('max_iterations', 'Não salvo')
        print(f"Número de iterações salvas: {num_iteracoes}")
        if max_iter_salvo != 'Não salvo' and num_iteracoes >= max_iter_salvo:
            print(">>> ALERTA: A otimização pode ter atingido o número máximo de iterações sem convergir! <<<")

        for key in iter_keys:
            grupo = hf[key]
            num_iter = int(key.split('_')[1])
            dados_iteracao.append({
                'iteracao': num_iter,
                'Objective_Function': grupo.attrs.get('Objective_Function'),
                'Tikhonov': grupo.attrs.get('Tikhonov'),
                'step_size': grupo.attrs.get('step_size')
            })

    df_iter = pd.DataFrame(dados_iteracao).sort_values('iteracao')
    
    # --- Plot da Evolução ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Diagnóstico da Otimização para Lambda = {lambda_problematico:.1e}', fontsize=16)

    # Gráfico 1: Função Objetivo
    axs[0].plot(df_iter['iteracao'].to_numpy(), df_iter['Objective_Function'].to_numpy(), 'r-') # <--- CORREÇÃO AQUI
    axs[0].set_ylabel('Função Objetivo')
    axs[0].set_title('Evolução da Função Objetivo')
    axs[0].grid(True)

    # Gráfico 2: Tikhonov
    axs[1].plot(df_iter['iteracao'].to_numpy(), df_iter['Tikhonov'].to_numpy(), 'b-') # <--- CORREÇÃO AQUI
    axs[1].set_ylabel('Valor de Tikhonov')
    axs[1].set_title('Evolução do Termo de Tikhonov')
    axs[1].grid(True)
    
    # Gráfico 3: Step Size
    axs[2].plot(df_iter['iteracao'].to_numpy(), df_iter['step_size'].to_numpy(), 'g-') # <--- CORREÇÃO AQUI
    axs[2].set_ylabel('Tamanho do Passo (Step Size)')
    axs[2].set_title('Evolução do Tamanho do Passo')
    axs[2].set_xlabel('Iteração')
    axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em '{caminho_completo}'")
except Exception as e:
    print(f"Ocorreu um erro: {e}")

try:
    with h5py.File(caminho_completo, 'r') as hf:
        # 1. Pega o atributo 'Morozov' da raiz do arquivo
        morozov_val = hf.attrs.get('Morozov', 'Não encontrado')
        
        # 2. Encontra todas as chaves de iteração
        iter_keys = [k for k in hf.keys() if k.startswith('iteration_')]
        
        if not iter_keys:
            print("Nenhuma iteração foi encontrada neste arquivo.")
        else:
            # 3. Encontra o número da última iteração
            last_iter_num = max([int(k.split('_')[1]) for k in iter_keys])
            print(f"Última iteração encontrada: {last_iter_num}")
            
            last_iter_group = hf[f'iteration_{last_iter_num}']
            
            # Pega os valores dos ATRIBUTOS da iteração
            obj_func_final = last_iter_group.attrs.get('Objective_Function', 'Não encontrado')
            tikhonov_final = last_iter_group.attrs.get('Tikhonov', 'Não encontrado')
            
            # Carrega os dados do DATASET 'heat_flux'
            heat_flux_final = last_iter_group['heat_flux'][:] if 'heat_flux' in last_iter_group else None

            # --- Impressão dos Resultados ---
            print("\n--- Atributo Global do Arquivo ---")
            print(f"Critério de Morozov: {morozov_val}")

            print("\n--- Valores Finais da Última Iteração (Atributos) ---")
            print(f"Função Objetivo (final): {obj_func_final}")
            print(f"Valor de Tikhonov (final): {tikhonov_final}")
            
            if heat_flux_final is not None:
                print("\n--- Dados Finais (Dataset 'heat_flux') ---")
                print(f"Shape (formato): {heat_flux_final.shape}")
                print(f"Primeiros valores: {heat_flux_final}")

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em '{caminho_completo}'")
except Exception as e:
    print(f"Ocorreu um erro: {e}")