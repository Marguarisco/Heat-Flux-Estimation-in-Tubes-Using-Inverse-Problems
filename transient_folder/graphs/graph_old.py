import pandas as pd
import matplotlib.pyplot as plt

time = 1000
N = 2
max_iterations = 1000000

# Carregar os dados
base_path = "Transient Periodic File/data"
real_flux = pd.read_csv(base_path + f"/heat_flux_real/heat_flux_real_9_80_{time}.csv")  # nt x ntheta
estimated_flux = pd.read_csv(f'Transient Periodic File/output/transient_heat_flux_9_80_{time}_{N}_{max_iterations}.csv')  # nt x ntheta

real_temperature = pd.read_csv(base_path+ f'/temperature/Temperature_Boundary_External_9_80_{time}.csv')  # 1D array
estimated_temperature = pd.read_csv(f'Transient Periodic File/output/transient_temperature_9_80_{time}_{N}_{max_iterations}.csv')  # 1D array

# Converter para NumPy arrays
real_flux = real_flux.to_numpy()
estimated_flux = estimated_flux.to_numpy()

reduction_factor = (real_flux.shape[1]//estimated_flux.shape[1])

values_real_temperature = real_temperature.iloc[-1, ::reduction_factor].to_numpy()
values_estimated_temperature = estimated_temperature.iloc[:, -1].to_numpy()

theta_space = range(len(values_real_temperature))

flux_diff = real_flux[:, ::reduction_factor] - estimated_flux

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

print(
    f"Real Flux: min={real_flux.min():.2f}, max={real_flux.max():.2f} | "
    f"Estimated Flux: min={estimated_flux.min():.2f}, max={estimated_flux.max():.2f} | "
    f"Flux Difference: min={flux_diff.min():.2f}, max={flux_diff.max():.2f}"
)

# Fluxo real
im0 = axs[0, 0].imshow(real_flux, aspect='auto', cmap='viridis', vmin=0, vmax=10000)
axs[0, 0].set_title("Real Heat Flux")
axs[0, 0].set_xlabel("Theta")
axs[0, 0].set_ylabel("Time")
fig.colorbar(im0, ax=axs[0, 0])

# Fluxo estimado
im1 = axs[0, 1].imshow(estimated_flux, aspect='auto', cmap='viridis', vmin=0, vmax=10000)
axs[0, 1].set_title("Estimated Heat Flux")
axs[0, 1].set_xlabel("Theta")
axs[0, 1].set_ylabel("Time")
fig.colorbar(im1, ax=axs[0, 1])

# Diferença
vmax = abs(flux_diff).max()
im2 = axs[1, 0].imshow(flux_diff, aspect='auto', cmap='seismic', vmin=-3000, vmax=3000)
axs[1, 0].set_title("Difference (Real - Estimated) Heat Flux")
axs[1, 0].set_xlabel("Theta")
axs[1, 0].set_ylabel("Time")
fig.colorbar(im2, ax=axs[1, 0])

# Temperatura
axs[1, 1].plot(theta_space, values_real_temperature, label='Real Temperature')
axs[1, 1].plot(theta_space, values_estimated_temperature, label='Estimated Temperature')
axs[1, 1].set_title("Temperature vs Theta")
axs[1, 1].set_xlabel("Theta")
axs[1, 1].set_ylabel("Temperature")
axs[1, 1].set_ylim(300, 350)
axs[1, 1].legend()
axs[1, 1].grid(True)


plt.tight_layout()
plt.show()