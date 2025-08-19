import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker

def graph_solo(T, dt, method):
    """
    Gera e salva gráficos do perfil de temperatura de uma única simulação
    numérica, com limites do eixo Y fixos para melhor comparação.

    Args:
        T_numerico (np.ndarray): Array 3D com os resultados da temperatura.
        dt (float): O passo de tempo utilizado na simulação (em segundos).
        method (str): Nome do método para usar nos títulos e nomes de arquivo.
    """
    # --- 1. Configuração da Malha e Parâmetros ---
    num_passos_tempo, num_angulos, num_raios = T.shape
    tempos_para_salvar = [0, 1, 5, 10, 50, 100, 599]
    R = np.linspace(0.1, 0.15, num_raios)
    T_init = T[0, -1, :]

    # --- 2. Loop para Geração dos Gráficos ---
    for tempo_s in tempos_para_salvar:
        indice_quadro = int(tempo_s / dt)
        if indice_quadro >= num_passos_tempo:
            print(f"Aviso: O tempo {tempo_s}s está além da simulação. Gráfico não gerado.")
            continue
        print(f"Gerando gráfico para o método '{method}' no tempo = {tempo_s}s...")
        T_estimativa = T[indice_quadro, -1, :]

        # --- 3. Lógica de Plotagem (Gráfico Único) ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(R, T_init, 'k--', linewidth=1.5, label='Temp. Inicial')
        ax.plot(R, T_estimativa, 'bo-', markersize=4, label=f'Resultado ({method})')
        ax.set_title(f'Perfil de Temperatura ({method}) - Tempo = {tempo_s:.1f} s')
        ax.set_xlabel('Posição Radial (m)')
        ax.set_ylabel('Temperatura (K)')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # --- 4. Salvar a Figura ---
        path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
        nome_arquivo = path + f'solo_{method}_tempo_{int(tempo_s)}s.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\nGráficos para o método '{method}' salvos com sucesso!")

def gif_solo(T, dt, method):
    """
    Cria e salva uma animação GIF mostrando a evolução da temperatura,
    com o eixo Y fixo para uma visualização estável.

    Args:
        T (np.ndarray): Array 3D com os resultados da temperatura.
        dt (float): O passo de tempo da simulação (em segundos).
        method (str): Nome do método para usar nos títulos e nome do arquivo.
    """
    print(f"Iniciando a criação da animação para o método '{method}'...")

    # --- 1. Configuração ---
    PASSO_ANIMACAO = 40
    num_passos_tempo_total = T.shape[0]
    num_quadros_final = num_passos_tempo_total // PASSO_ANIMACAO
    print(f"A animação terá {num_quadros_final} quadros.")

    # --- 2. Preparar a Figura e os Dados Iniciais ---
    num_raios = T.shape[2]
    R = np.linspace(0.1, 0.15, num_raios)
    T_init = T[0, -1, :]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_tight_layout(True)

    # --- 3. Função de Atualização da Animação ---
    def update(i):
        passo_de_tempo_atual = i * PASSO_ANIMACAO
        tempo_em_segundos = passo_de_tempo_atual * dt
        T_estimativa = T[passo_de_tempo_atual, -1, :]
        ax.clear()
        ax.plot(R, T_init, 'k--', linewidth=1.5, label='Temp. Inicial')
        ax.plot(R, T_estimativa, 'bo-', markersize=4, label=f'Resultado ({method})')
        ax.set_title(f'Evolução da Temperatura ({method}) - Tempo = {tempo_em_segundos:.1f} s')
        ax.set_xlabel('Posição Radial (m)')
        ax.set_ylabel('Temperatura (K)')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        if (i + 1) % 25 == 0 or i == num_quadros_final - 1:
            print(f"Processando quadro {i + 1}/{num_quadros_final}...")

    # --- 4. Criar e Salvar a Animação ---
    ani = animation.FuncAnimation(fig, update, frames=num_quadros_final, interval=50)
    print("\nSalvando o GIF...")
    path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
    nome_arquivo = path + f'solo_{method}.gif'
    ani.save(nome_arquivo, writer='pillow', fps=15, dpi=100)
    plt.close(fig)
    print(f"\nAnimação salva com sucesso em '{nome_arquivo}'!")

def graph_temp_time(T, method):
    temp_ponto_especifico = T[:, 0, -1] # [todos os tempos, primeiro angulo, ultimo raio]

    plt.figure(figsize=(12, 6))
    plt.plot(temp_ponto_especifico)
    plt.title('Evolução da Temperatura no Ponto Externo (r_max, θ=0)')
    plt.xlabel('Passo de Tempo da Simulação')
    plt.ylabel('Temperatura (K)')
    plt.grid(True)

    path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
    plt.savefig(path + f'grafico_temperatura_tempo_{method}.jpg')
    plt.close()
    
    print(f"Gráfico salvo com sucesso")

# Graphs made for comparison between analitic data
def graph_comparison_analitic(T, dt, method):
    dados_analitico = pd.read_csv('C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv', header = None)

    num_passos_tempo, num_angulos, num_raios = T.shape

    tempos_para_salvar = [0, 1, 5, 10, 50, 100, 599]

    R = np.linspace(0.1, 0.15, num_raios) # Malha Radial

    plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan = 2)
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    T_init = T[0,-1,:]

    for tempo_s in tempos_para_salvar:
        # Converte o tempo em segundos para o índice do quadro (frame)
        indice_quadro = int(tempo_s / dt)
        
        # Medida de segurança: verifica se o índice calculado existe na sua simulação
        if indice_quadro >= num_passos_tempo:
            print(f"Aviso: O tempo {tempo_s}s (quadro {indice_quadro}) está além da duração da simulação. Gráfico não gerado.")
            continue # Pula para o próximo tempo da lista

        print(f"Gerando gráfico para o tempo = {tempo_s}s (quadro = {indice_quadro})...")
        
        # --- 3. Lógica de Plotagem (quase idêntica à sua) ---
        T_analitico = dados_analitico.iloc[:, indice_quadro].to_numpy()
        T_estimativa = T[indice_quadro, -1, :]
        T_diff = T_estimativa - T_analitico

        # Cria uma nova figura para cada tempo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        fig.set_tight_layout(True)
        
        # Plot da Comparação
        ax1.plot(R, T_init, 'k--', linewidth=1.5, label='Temp. Inicial')
        ax1.plot(R, T_estimativa, 'bo-', markersize=4, label="Estimativa")
        ax1.plot(R, T_analitico, 'r^--', markersize=5, label="Analítico")
        ax1.set_title(f'Comparação de Temperaturas (Tempo = {tempo_s:.1f} s)')
        ax1.set_xlabel('Posição Radial (m)'); ax1.set_ylabel('Temperatura (K)')
        ax1.legend(); ax1.grid(True, linestyle=':')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Plot da Diferença
        ax2.plot(R, T_diff, 'ko-', markersize=4)
        ax2.set_title('Diferença (Estimativa - Analítico)')
        ax2.set_xlabel('Posição Radial (m)'); ax2.set_ylabel('Temperatura (K)')
        ax2.set_ylim(-0.1, 0.1); ax2.axhline(0, color='k', lw=0.5)
        ax2.grid(True, linestyle=':')
        
        # --- 4. Salvar a Figura ---
        # Cria um nome de arquivo dinâmico para cada imagem. Usar PNG é melhor para gráficos.
        path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
        nome_arquivo = path + f'comparacao_tempo_{method}_{int(tempo_s)}s.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        
        # Fecha a figura para liberar memória antes de criar a próxima
        plt.close(fig)

def gif_comparison_analitic(T, method):
    """
    Cria e salva uma animação comparando a solução numérica com a analítica.
    """
    print("Carregando dados e preparando a animação...")

    dados_analitico_path = 'C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv'
    
    # --- 1. Carregar Dados ---
    try:
        dados_analitico = pd.read_csv(dados_analitico_path, header=None)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{dados_analitico_path}'")
        return

    # --- 2. Otimização Principal: Reduzir o Número de Quadros ---
    # Defina de quantos em quantos passos você quer salvar um quadro.
    # Valores maiores = mais rápido, arquivo menor, animação menos fluida.
    PASSO_ANIMACAO = 50  # Sugestão: comece com 40 (6000/40 = 150 quadros)

    # Lógica para evitar o erro de 'out of bounds'
    num_passos_tempo_total = min(T.shape[0], dados_analitico.shape[1])
    
    # Calcula o número final de quadros para a animação
    num_quadros_final = num_passos_tempo_total // PASSO_ANIMACAO
    
    print(f"A animação terá {num_quadros_final} quadros (1 a cada {PASSO_ANIMACAO} passos de tempo).")

    # --- 3. Preparar a Figura e os Dados ---
    num_raios = T.shape[2]
    R = np.linspace(0.1, 0.15, num_raios)
    T_init = T[0, -1, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.set_tight_layout(True)

    # --- 4. Função de Atualização da Animação ---
    def update(i):
        passo_de_tempo_atual = i * PASSO_ANIMACAO
        
        T_analitico = dados_analitico.iloc[:, passo_de_tempo_atual].to_numpy()
        T_estimativa = T[passo_de_tempo_atual, -1, :]
        T_diff = T_estimativa - T_analitico

        # Atualiza os subplots
        ax1.clear()
        ax1.plot(R, T_init, 'k--', label='Temp. Inicial')
        ax1.plot(R, T_estimativa, 'bo-', label="Estimativa")
        ax1.plot(R, T_analitico, 'r^--', label="Analítico")
        ax1.set_title(f'Comparação (Tempo = {passo_de_tempo_atual / 10.0:.1f} s)')
        ax1.set_xlabel('Posição Radial (m)'); ax1.set_ylabel('Temperatura (K)')
        ax1.legend(); ax1.grid(True, linestyle=':')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        ax2.clear()
        ax2.plot(R, T_diff, 'ko-')
        ax2.set_title('Diferença (Estimativa - Analítico)'); 
        ax2.set_xlabel('Posição Radial (m)'); ax2.set_ylabel('Temperatura (K)')
        ax2.set_ylim(-0.1, 0.1); ax2.axhline(0, color='k', lw=0.5)
        ax2.grid(True, linestyle=':')
        
        if (i + 1) % 20 == 0:
            print(f"Processando quadro {i + 1}/{num_quadros_final}...")

    # --- 5. Criar e Salvar a Animação como GIF ---
    ani = animation.FuncAnimation(fig, update, frames=num_quadros_final, interval=50, blit=False)


    print("Salvando como GIF usando o escritor 'pillow' (pode levar um momento)...")
    # dpi (resolução) e fps (quadros por segundo) controlam a aparência final.
    path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
    ani.save(path + f'comparacao_{method}.gif', writer='pillow', fps=15, dpi=100)
    print("\nAnimação salva com sucesso em 'comparacao_animada.gif'!")


# Graphs made for ADI full analysis
def heat_map_2d(T, dt, method):
    num_passos_tempo, num_angulos, num_raios = T.shape

    R = np.linspace(0.1, 0.15, num_raios) # Malha Radial
    Theta = np.linspace(0, 2 * np.pi, num_angulos, endpoint = False) # Malha angular
    I, J = np.meshgrid(R, Theta) 

    X = I * np.cos(J)
    Y = I * np.sin(J)
    X = np.vstack((X, X[0,:])) # Adicionando um valor igual a 0° p/ plot
    Y = np.vstack((Y, Y[0,:])) # Adicionando um valor igual a 0° p/ plot

    # Melhoria: usa o min/max real dos dados para a escala de cores
    temp_min = T.min()
    temp_max = T.max()
    niveis_de_cor = np.linspace(temp_min, temp_max, 100)

    ultimo_quadro = num_passos_tempo - 1

    # Lista dos índices dos quadros que queremos salvar
    indices_para_salvar = [
        0,                                  # Início (t=0)
        ultimo_quadro // 4,                 # 1/4 do tempo
        ultimo_quadro // 2,                 # Metade do tempo
        3 * ultimo_quadro // 4,             # 3/4 do tempo
        ultimo_quadro                       # Fim (t_max)
    ]
    # Remove duplicados caso a simulação seja muito curta
    indices_para_salvar = sorted(list(set(indices_para_salvar))) 

    # --- 3. Loop para Gerar e Salvar os Gráficos ---
    for n in indices_para_salvar:
        # Calcula o tempo em segundos para o título
        tempo_em_segundos = n * dt
        print(f"Gerando gráfico para o tempo = {tempo_em_segundos:.2f} s (quadro {n})...")
        
        # Prepara os dados de temperatura para o quadro 'n'
        T_quadro = np.vstack((T[n], T[n, 0]))
        
        # Cria uma nova figura para cada imagem
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plota o heatmap
        contour = ax.contourf(X, Y, T_quadro, levels=niveis_de_cor, cmap='inferno')
        fig.colorbar(contour, ax=ax, label="Temperatura (K)")
        
        # Configura os rótulos e título
        ax.set_title(f'Distribuição de Temperatura (Tempo = {tempo_em_segundos:.2f} s)')
        ax.set_xlabel('Posição X (m)')
        ax.set_ylabel('Posição Y (m)')
        ax.axis('equal')
        fig.set_tight_layout(True)

        # Salva a figura em um arquivo JPEG de alta qualidade
        path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
        nome_arquivo = path + f'heatmap_tempo_{method}_{tempo_em_segundos:.0f}s.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        # Fecha a figura para liberar memória
        plt.close(fig)

    print("\nTodos os gráficos foram salvos com sucesso!")

def gif_heatmap(T, dt, method):
    """
    Cria e salva uma animação GIF da distribuição de temperatura 2D.
    """
    print("Iniciando a criação do GIF do heatmap...")

    # --- 1. Preparar os Dados e a Malha ---
    num_passos_tempo, num_angulos, num_raios = T.shape

    # Malha Polar
    R = np.linspace(0.1, 0.15, num_raios)
    Theta = np.linspace(0, 2 * np.pi, num_angulos, endpoint=False)
    
    # Converter para Malha Cartesiana para o plot
    I, J = np.meshgrid(R, Theta)
    X = I * np.cos(J)
    Y = I * np.sin(J)
    
    # Adicionar pontos extras para "fechar" o círculo e evitar falhas visuais
    X = np.vstack((X, X[0, :]))
    Y = np.vstack((Y, Y[0, :]))

    # --- 2. Otimização: Definir o Pulo de Quadros ---
    # Usaremos a mesma lógica do seu código: atualizar a cada 100 passos
    PASSO_ANIMACAO = 100
    num_quadros_final = num_passos_tempo // PASSO_ANIMACAO
    print(f"A animação terá {num_quadros_final} quadros (1 a cada {PASSO_ANIMACAO} passos de tempo).")

    # --- 3. Configurar a Figura e a Escala de Cores ---
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Definir uma escala de cores fixa é CRUCIAL para a animação.
    # Use os valores mínimo e máximo de toda a simulação.
    temp_min = T.min()
    temp_max = T.max()
    niveis_de_cor = np.linspace(temp_min, temp_max, 100)

    # --- 4. Definir a Função de Atualização ---
    # Esta função será chamada para desenhar cada quadro do GIF.
    # 'i' é o índice do quadro da animação (de 0 até num_quadros_final - 1).
    def update(i):
        # Mapeia o quadro da animação para o passo de tempo original
        n = i * PASSO_ANIMACAO
        
        # Limpa o eixo para o novo desenho
        ax.clear()
        
        # Prepara a matriz de temperatura para fechar o círculo
        T_quadro = np.vstack((T[n], T[n, 0]))
        
        # Desenha o heatmap do quadro atual
        contour = ax.contourf(X, Y, T_quadro, levels=niveis_de_cor, cmap='inferno')
        
        # Reconfigura os rótulos e o título para o quadro atual
        ax.set_title(f'Distribuição de Temperatura (Tempo = {n * dt:.2f} s)')
        ax.set_xlabel('Posição X (m)')
        ax.set_ylabel('Posição Y (m)')
        ax.axis('equal') # Garante que o círculo não fique distorcido
        
        # A colorbar precisa ser adicionada a cada quadro se 'ax.clear()' for usado
        # Se isso ficar lento, podemos usar métodos mais avançados, mas este é o mais simples.
        # fig.colorbar(contour, ax=ax) # Descomente se precisar da colorbar em cada quadro
        
        if (i + 1) % 10 == 0:
            print(f"Processando quadro {i + 1}/{num_quadros_final}...")
        
        return contour,

    # --- 5. Criar e Salvar a Animação ---
    print("Criando o objeto de animação...")
    ani = animation.FuncAnimation(fig, update, frames=num_quadros_final, interval=100, blit=False)
    
    # Adicionar uma única colorbar que vale para toda a animação
    # Criamos um plot estático apenas para gerar a colorbar
    contour_static = ax.contourf(X, Y, np.vstack((T[0], T[0,0])), levels=niveis_de_cor, cmap='inferno')
    fig.colorbar(contour_static, ax=ax, label="Temperatura (K)")
    fig.set_tight_layout(True)
    
    print("Salvando o GIF (pode levar um momento)...")
    # Usamos o 'writer' pillow e um DPI razoável para acelerar
    path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
    ani.save(path + f'heatmap_{method}.gif', writer='pillow', fps=10, dpi=100)
    print("\nAnimação salva com sucesso em 'heatmap_animado.gif'!")


# Comparison between the three methods
def graph_comparison_analitic_all(T_explicit, T_implicit, T_adi, dt):
    """
    Gera e salva gráficos comparando as soluções numéricas (Explícito, Implícito, ADI)
    com a solução analítica em instantes de tempo específicos.

    Args:
        T_explicit (np.ndarray): Array 3D com os resultados da temperatura do método explícito.
                                 Dimensões: (num_passos_tempo, num_angulos, num_raios)
        T_implicit (np.ndarray): Array 3D com os resultados da temperatura do método implícito.
                                 Dimensões: (num_passos_tempo, num_angulos, num_raios)
        T_adi (np.ndarray): Array 3D com os resultados da temperatura do método ADI.
                            Dimensões: (num_passos_tempo, num_angulos, num_raios)
        dt (float): O passo de tempo utilizado na simulação (em segundos).
    """
    # --- 1. Carregamento dos Dados Analíticos ---
    # Certifique-se de que o caminho para o arquivo CSV está correto.
    try:
        dados_analitico = pd.read_csv('C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv', header=None)
    except FileNotFoundError:
        print("Erro: O arquivo 'dados_2d_Bruno_v.2.csv' não foi encontrado. Verifique o caminho.")
        return

    # --- 2. Configuração da Malha e Parâmetros ---
    # As dimensões são extraídas do primeiro array de entrada (assumindo que todas são iguais)
    num_passos_tempo, num_angulos, num_raios = T_explicit.shape

    # Lista de tempos em segundos para os quais os gráficos serão gerados
    tempos_para_salvar = [0, 1, 5, 10, 50, 100, 599]

    # Definição da malha radial (deve corresponder à sua simulação)
    R = np.linspace(0.1, 0.15, num_raios)

    # --- 3. Loop para Geração dos Gráficos ---
    for tempo_s in tempos_para_salvar:
        # Converte o tempo em segundos para o índice do array
        indice_quadro = int(tempo_s / dt)

        # Medida de segurança: verifica se o índice existe na simulação
        if indice_quadro >= num_passos_tempo:
            print(f"Aviso: O tempo {tempo_s}s (quadro {indice_quadro}) está além da duração da simulação. Gráfico não gerado.")
            continue  # Pula para o próximo tempo

        print(f"Gerando gráfico para o tempo = {tempo_s}s (quadro = {indice_quadro})...")

        # --- 4. Extração dos Dados para o Tempo Atual ---
        T_analitico = dados_analitico.iloc[:, indice_quadro].to_numpy()
        T_init = T_explicit[0, -1, :] # A temperatura inicial é a mesma para todos

        # Estimativas de cada método no último ângulo (assumindo simetria)
        T_est_explicit = T_explicit[indice_quadro, -1, :]
        T_est_implicit = T_implicit[indice_quadro, -1, :]
        T_est_adi = T_adi[indice_quadro, -1, :]

        # Cálculo da diferença de cada método em relação ao analítico
        T_diff_explicit = T_est_explicit - T_analitico
        T_diff_implicit = T_est_implicit - T_analitico
        T_diff_adi = T_est_adi - T_analitico

        # --- 5. Lógica de Plotagem ---
        # Cria uma nova figura e subplots para cada tempo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
        fig.set_tight_layout(True)
        fig.suptitle(f'Comparação de Métodos vs. Analítico (Tempo = {tempo_s:.1f} s)', fontsize=16, y=1.02)

        # Plot 1: Comparação das Temperaturas
        ax1.plot(R, T_init, 'k--', linewidth=1.5, label='Temp. Inicial')
        ax1.plot(R, T_analitico, 'r^--', markersize=5, label="Analítico")
        ax1.plot(R, T_est_explicit, 'o-', color='blue', markersize=4, label="Explícito")
        ax1.plot(R, T_est_implicit, 's-', color='green', markersize=4, label="Implícito")
        ax1.plot(R, T_est_adi, 'd-', color='purple', markersize=4, label="ADI")
        ax1.set_title('Comparação de Perfis de Temperatura')
        ax1.set_xlabel('Posição Radial (m)')
        ax1.set_ylabel('Temperatura (K)')
        ax1.legend()
        ax1.grid(True, linestyle=':')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # Plot 2: Diferença (Estimativas - Analítico)
        ax2.plot(R, T_diff_explicit, 'o-', color='blue', markersize=4, label="Erro Explícito")
        ax2.plot(R, T_diff_implicit, 's-', color='green', markersize=4, label="Erro Implícito")
        ax2.plot(R, T_diff_adi, 'd-', color='purple', markersize=4, label="Erro ADI")
        ax2.set_title('Diferença (Numérico - Analítico)')
        ax2.set_xlabel('Posição Radial (m)')
        ax2.set_ylabel('Diferença de Temperatura (K)')
        ax2.set_ylim(-0.1, 0.1)  # Ajuste este limite se necessário
        ax2.axhline(0, color='k', lw=0.7)
        ax2.legend()
        ax2.grid(True, linestyle=':')

        # --- 6. Salvar a Figura ---
        # Define o caminho e um nome de arquivo dinâmico para cada imagem
        path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
        nome_arquivo = path + f'comparacao_geral_tempo_{int(tempo_s)}s.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')

        # Fecha a figura para liberar memória
        plt.close(fig)

    print("\nTodos os gráficos foram gerados e salvos com sucesso!")

def gif_comparison_analitic_all(T_explicit, T_implicit, T_adi, dt):
    """
    Cria e salva uma animação GIF comparando as soluções numéricas (Explícito,
    Implícito, ADI) com a solução analítica ao longo do tempo.

    Args:
        T_explicit (np.ndarray): Array 3D com os resultados da temperatura do método explícito.
        T_implicit (np.ndarray): Array 3D com os resultados da temperatura do método implícito.
        T_adi (np.ndarray): Array 3D com os resultados da temperatura do método ADI.
        dt (float): O passo de tempo utilizado na simulação (em segundos).
    """
    print("Iniciando a criação da animação comparativa...")

    # --- 1. Carregar Dados Analíticos ---
    dados_analitico_path = 'C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv'
    try:
        dados_analitico = pd.read_csv(dados_analitico_path, header=None)
    except FileNotFoundError:
        print(f"Erro: Arquivo analítico não encontrado em '{dados_analitico_path}'")
        return

    # --- 2. Otimização e Configuração da Animação ---
    # Define de quantos em quantos passos de tempo um quadro será salvo no GIF.
    # Valores maiores = geração mais rápida, arquivo menor, mas animação menos fluida.
    PASSO_ANIMACAO = 40  # Salva 1 quadro a cada 40 passos de tempo

    # Garante que a animação não tente acessar um índice inexistente
    num_passos_tempo_total = min(T_explicit.shape[0], dados_analitico.shape[1])
    num_quadros_final = num_passos_tempo_total // PASSO_ANIMACAO

    print(f"A animação terá {num_quadros_final} quadros (1 a cada {PASSO_ANIMACAO} passos de tempo).")

    # --- 3. Preparar a Figura e os Dados Iniciais ---
    num_raios = T_explicit.shape[2]
    R = np.linspace(0.1, 0.15, num_raios)
    T_init = T_explicit[0, -1, :] # Condição inicial é a mesma para todos

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.set_tight_layout(True)

    # --- 4. Função de Atualização (chamada para cada quadro) ---
    def update(i):
        # Calcula o passo de tempo real correspondente ao quadro 'i'
        passo_de_tempo_atual = i * PASSO_ANIMACAO
        tempo_em_segundos = passo_de_tempo_atual * dt
        
        # Extrai os dados para o instante atual
        T_analitico = dados_analitico.iloc[:, passo_de_tempo_atual].to_numpy()
        T_est_explicit = T_explicit[passo_de_tempo_atual, -1, :]
        T_est_implicit = T_implicit[passo_de_tempo_atual, -1, :]
        T_est_adi = T_adi[passo_de_tempo_atual, -1, :]
        
        T_diff_explicit = T_est_explicit - T_analitico
        T_diff_implicit = T_est_implicit - T_analitico
        T_diff_adi = T_est_adi - T_analitico

        # Limpa os eixos para desenhar o novo quadro
        ax1.clear()
        ax2.clear()

        # Atualiza o subplot 1 (Comparação de Temperaturas)
        ax1.plot(R, T_init, 'k--', linewidth=1.5, label='Temp. Inicial')
        ax1.plot(R, T_analitico, 'r^--', markersize=5, label="Analítico")
        ax1.plot(R, T_est_explicit, 'o-', color='blue', markersize=4, label="Explícito")
        ax1.plot(R, T_est_implicit, 's-', color='green', markersize=4, label="Implícito")
        ax1.plot(R, T_est_adi, 'd-', color='purple', markersize=4, label="ADI")
        ax1.set_title(f'Comparação de Métodos (Tempo = {tempo_em_segundos:.1f} s)')
        ax1.set_xlabel('Posição Radial (m)'); ax1.set_ylabel('Temperatura (K)')
        ax1.legend(); ax1.grid(True, linestyle=':')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        
        # Atualiza o subplot 2 (Diferença)
        ax2.plot(R, T_diff_explicit, 'o-', color='blue', markersize=4)
        ax2.plot(R, T_diff_implicit, 's-', color='green', markersize=4)
        ax2.plot(R, T_diff_adi, 'd-', color='purple', markersize=4)
        ax2.set_title('Diferença (Numérico - Analítico)')
        ax2.set_xlabel('Posição Radial (m)'); ax2.set_ylabel('Diferença (K)')
        ax2.set_ylim(-0.1, 0.1)
        ax2.axhline(0, color='k', lw=0.7)
        ax2.grid(True, linestyle=':')

        # Imprime o progresso no console
        if (i + 1) % 25 == 0 or i == num_quadros_final - 1:
            print(f"Processando quadro {i + 1}/{num_quadros_final}...")

    # --- 5. Criar e Salvar a Animação ---
    ani = animation.FuncAnimation(fig, update, frames=num_quadros_final, interval=50, blit=False)

    print("\nSalvando o GIF (isso pode levar alguns minutos)...")
    path = 'C:/Users/marce/Desktop/TCC/Direct Problem/Results/'
    nome_arquivo = path + 'comparacao_geral_animada.gif'
    
    # Usa o 'writer' pillow para salvar como .gif
    ani.save(nome_arquivo, writer='pillow', fps=15, dpi=100)
    
    # Fecha a figura após salvar
    plt.close(fig)
    print(f"\nAnimação salva com sucesso em '{nome_arquivo}'!")
