import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


dados = np.load('C:/Users/marce/Desktop/TCC/Direct Problem/Results/resultados_implicit_completo_25_3_6000.npz')
dados_analitico = pd.read_csv('C:/Users/marce/Desktop/Old Code Versions/Dados_comparação/dados_2d_Bruno_v.2.csv', header = None)

T = dados['temperaturas']

print("Arquivo carregado com sucesso!")
print(f"Forma do array de temperaturas: {T.shape}")

# Pega o número de passos de tempo, ângulos e raios a partir da forma do array
num_passos_tempo, num_angulos, num_raios = T.shape


# --- 4. Exemplo de Visualização ---

# Exemplo A: Plotar a temperatura no ponto mais externo (r_outer) e no primeiro ângulo (theta=0) ao longo do tempo.
temp_ponto_especifico = T[:, 0, -1] # [todos os tempos, primeiro angulo, ultimo raio]



R = np.linspace(0.1, 0.15, num_raios) # Malha Radial
Theta = np.linspace(0, 2 * np.pi, num_angulos, endpoint = False) # Malha angular
I, J = np.meshgrid(R, Theta) 

X = I * np.cos(J)
Y = I * np.sin(J)
X = np.vstack((X, X[0,:])) # Adicionando um valor igual a 0° p/ plot
Y = np.vstack((Y, Y[0,:])) # Adicionando um valor igual a 0° p/ plot



plt.figure(figsize=(12, 6))
plt.plot(temp_ponto_especifico)
plt.title('Evolução da Temperatura no Ponto Externo (r_max, θ=0)')
plt.xlabel('Passo de Tempo da Simulação')
plt.ylabel('Temperatura (°K)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan = 2)
ax2 = plt.subplot2grid((1, 3), (0, 2))

T_init = T[0,-1,:]

for i in range(num_passos_tempo):
    T_analitico = dados_analitico.iloc[:,i].to_numpy()
    T_diff = T[i, -1, :] - T_analitico

    ax1.clear() #Limpar os dados
    ax1.plot(R[::], T_init, 'ko-', markersize=5, label = 'Temperatura Inicial')
    ax1.plot(R[::], T[i,-1,:], 'bo-', markersize=3, label = "Estimativa")
    ax1.plot(R[::], T_analitico,'ro--', markersize=3, label = "Analítico")
    ax1.set_title(f'Temperaturas em {i*0.1:.1f} s') # Título
    ax1.set_xlabel('Posição')  # Legenda do eixo x
    ax1.set_ylabel('Temperatura (K)')  # Legenda do eixo y
    ax1.legend()
    plt.tight_layout()
    plt.pause(0.000001)

    ax2.clear() #Limpar os dados
    ax2.plot(R[::], T_diff,'ko-', markersize=5, label = "Diferenças")
    ax2.set_title('Diferença das temperaturas') # Título
    ax2.set_xlabel('Posição na Barra')  # Legenda do eixo x
    ax2.set_ylabel('Diferença de temperaturas (K)')  # Legenda do eixo y
    ax2.set_ylim(-0.1, 0.1)
    ax2.legend()
