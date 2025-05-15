import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

time = 1100

values_real_temperature = pd.read_csv(f'Transient File/data/temperature/Temperature_Boundary_External_9_80_{time}.csv') # time x 80
values_estimated_temperature = pd.read_csv(f'Transient File/output/transient_temperature_9_80_{time}.csv') # time x 20

values_real_q = pd.read_csv(f'Transient File/data/heat_flux_real_{time}.csv')
values_estimated_q = pd.read_csv(f'Transient File/output/transient_heat_flux_9_80_{time}.csv')

values_real_temperature = values_real_temperature.iloc[-1, ::4].to_numpy()
values_estimated_temperature = values_estimated_temperature.iloc[:, -1].to_numpy()

theta_space = range(len(values_real_temperature))

if time == 1200:
    values_real_q = values_real_q.iloc[::4, -1].to_numpy()
    space = theta_space
    space_label = 'Theta Space'

else:
    values_real_q = values_real_q.iloc[:, -1].to_numpy()
    space = range(time)
    space_label = 'Time Space'

values_estimated_q = values_estimated_q.iloc[:, -1].to_numpy()


# Plotar os dados
plt.figure(figsize=(10, 6))

# Primeiro gráfico: Comparação de heat flux real e estimado
plt.subplot(2, 1, 1)  # 2 linhas, 1 coluna, 1º gráfico
line_real_q, = plt.plot(space, values_real_q, label='Real Heat Flux', color='blue')
line_estimated_q, = plt.plot(space, values_estimated_q, label='Estimated Heat Flux', color='orange')

plt.title('Heat Flux Comparison')
plt.xlabel(f'{space_label}')
plt.ylabel('Heat Flux')
plt.legend()
plt.grid(True)

# Adicionar interatividade com mplcursors
cursor_q = mplcursors.cursor([line_real_q, line_estimated_q], hover=True)
cursor_q.connect("add", lambda sel: sel.annotation.set_text(
    f"{space_label}: {int(sel.target[0])}\nValue: {sel.target[1]:.2f}"
))

# Segundo gráfico: Comparação de temperatura real e estimada
plt.subplot(2, 1, 2)  # 2 linhas, 1 coluna, 2º gráfico
line_real_temp, = plt.plot(theta_space, values_real_temperature, label='Real Temperature', color='green')
line_estimated_temp, = plt.plot(theta_space, values_estimated_temperature, label='Estimated Temperature', color='red')

plt.title('Temperature Comparison')
plt.xlabel('Theta Space')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)

# Adicionar interatividade com mplcursors
cursor_temp = mplcursors.cursor([line_real_temp, line_estimated_temp], hover=True)
cursor_temp.connect("add", lambda sel: sel.annotation.set_text(
    f"Theta: {int(sel.target[0])}\nValue: {sel.target[1]:.2f}"
))

# Ajustar layout e mostrar o gráfico
plt.tight_layout()
plt.show()