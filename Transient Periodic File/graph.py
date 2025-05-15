import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

time = 1100

# Carregar os dados
base_path = "Transient Periodic File/data"
real_flux = pd.read_csv(base_path + "/heat_flux_real.csv")  # 80 x 1000
estimated_flux = pd.read_csv(f'Transient Periodic File/output/transient_heat_flux_9_80_{time}.csv')  # 20 x 1000

real_temperature = pd.read_csv(base_path+ f'/temperature/Temperature_Boundary_External_9_80_{time}.csv')  # 1D array
estimated_temperature = pd.read_csv(f'Transient Periodic File/output/transient_temperature_9_80_{time}.csv')  # 1D array

# Converter para NumPy arrays
real_flux = real_flux.to_numpy()
estimated_flux = estimated_flux.to_numpy()

values_real_temperature = real_temperature.iloc[-1, ::4].to_numpy()
values_estimated_temperature = estimated_temperature.iloc[:, -1].to_numpy()

theta_space = range(len(values_real_temperature))
# Criar a figura com layout 2x2
fig = plt.figure(figsize=(12, 9))

# Gráfico do fluxo de calor real
ax1 = plt.subplot(2, 2, 1)
im_real = ax1.imshow(real_flux, aspect='auto', cmap='viridis')
ax1.set_title("Real Heat Flux")
ax1.set_xlabel("Theta")
ax1.set_ylabel("Time")
cbar_real = plt.colorbar(im_real, ax=ax1)
cbar_real.set_label("Heat Flux")

# Gráfico do fluxo de calor estimado
ax2 = plt.subplot(2, 2, 2)
im_estimated = ax2.imshow(estimated_flux, aspect='auto', cmap='viridis')
ax2.set_title("Estimated Heat Flux")
ax2.set_xlabel("Theta")
ax2.set_ylabel("Time")
cbar_estimated = plt.colorbar(im_estimated, ax=ax2)
cbar_estimated.set_label("Heat Flux")


# Gráfico de temperatura
ax3 = plt.subplot(2, 1, 2)
ax3.plot(theta_space, values_real_temperature, label='Real Temperature')
ax3.plot(theta_space, values_estimated_temperature, label='Estimated Temperature')
ax3.set_title("Temperature vs Theta")
ax3.set_xlabel("Theta")
ax3.set_ylabel("Temperature")
ax3.legend()
ax3.grid(True)

# Ajustar layout
plt.tight_layout()
plt.show()