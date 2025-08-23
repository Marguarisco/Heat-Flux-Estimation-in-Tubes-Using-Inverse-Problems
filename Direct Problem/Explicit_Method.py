import numpy as np
import pandas as pd
import numba

@numba.jit(nopython=True, fastmath=True)
def explicit_method(max_time_steps, args):

    temperature_args, spacial_args, material_args, flux_args = args

    T_inner, T_outer, T_tube = temperature_args
    dr, radial_size, radial_space, dtheta, angular_size, angular_space = spacial_args
    thermal_conductivity, specific_heat, density = material_args
    heat_flux, h_conv = flux_args

    thermal_diffusivity = thermal_conductivity / (specific_heat * density)  

    dt = 0.99 * ((dr**2)*(dtheta**2))/(2 * thermal_diffusivity * ((dr**2)+(dtheta**2)))

    dt_all = 0.1

    dt = dt_all / np.ceil(dt_all / dt)

    current_temp = np.ones((angular_size, radial_size)) * T_tube

    vector = np.zeros((angular_size, radial_size), dtype=np.float64) # Right-hand side vector

    gamma_r  = thermal_diffusivity * dt / (2 * dr * radial_space)
    gamma_rr = thermal_diffusivity * dt / (dr ** 2)
    gamma_tt = thermal_diffusivity * dt / ((dtheta ** 2) * (radial_space ** 2))

    beta_0 = (2 * dr * heat_flux) / thermal_conductivity
    gamma_0 = (- gamma_r[0] + gamma_rr) * beta_0

    beta_I = 2 * dr * h_conv / thermal_conductivity
    gamma_I = (gamma_r[-1] + gamma_rr) * beta_I 

    T_ext_history = np.zeros((max_time_steps, angular_size, radial_size), dtype=np.float64)
    T_ext_history[0, :, :] = current_temp

    time_step = 1
    while time_step < max_time_steps:
        for _ in range(dt_all/dt):
            for j in range(angular_size):
                for i in range(radial_size):
                    prev_j = (j - 1) % angular_size
                    next_j = (j + 1) % angular_size

                    if i != 0 and i != radial_size - 1:
                        vector[j, i] = ((1 - (2 * gamma_rr) - (2 * gamma_tt[i])) * current_temp[j, i]) + \
                                ((- gamma_r[i] + gamma_rr) * current_temp[j, i - 1]) + \
                                ((+ gamma_r[i] + gamma_rr) * current_temp[j, i + 1]) + \
                                gamma_tt[i] * current_temp[prev_j, i] + \
                                gamma_tt[i] * current_temp[next_j, i]
                        
                    elif i == 0:
                        vector[j, i] = ((1 - (2 * gamma_rr) - (2 * gamma_tt[i])) * current_temp[j, i]) + \
                            2 * gamma_rr * current_temp[j, i + 1] + \
                            gamma_0[j] + \
                            gamma_tt[i] * current_temp[prev_j, i] + \
                            gamma_tt[i] * current_temp[next_j, i]

                    elif i == radial_size - 1:
                        vector[j, i] = ((1 - (2 * gamma_rr) - (2 * gamma_tt[i]) - gamma_I) * current_temp[j, i]) + \
                            (2 * gamma_rr * current_temp[j, i - 1]) + \
                            (gamma_I * T_outer) + \
                            gamma_tt[i] * current_temp[prev_j, i] + \
                            gamma_tt[i] * current_temp[next_j, i]
            
            current_temp = np.copy(vector)    
            
        T_ext_history[time_step, :, :] = current_temp
        time_step += 1

    return T_ext_history, dt

if __name__ == '__main__':
    max_time_steps = 6000

    radial_size = 9
    angular_size = 80

    r_inner, r_outer = 0.1, 0.15

    dr = (r_outer - r_inner) / (radial_size - 1)  
    dtheta = 2 * np.pi / angular_size  

    radial_space = np.linspace(r_inner, r_outer, radial_size)
    angular_space = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)
    

    T_inner, T_outer, T_tube = 300.0, 300.0, 300.0  

    h_conv = 25.0
    heat_flux = ((-2000.0) * (angular_space / np.pi) ** 2) + 2000.0  

    thermal_conductivity = 201.0  
    specific_heat = 900.0  
    density = 2700.0 

    spacial_args = (dr,  radial_size, radial_space, dtheta, angular_size, angular_space)
    temperature_args = (T_inner, T_outer, T_tube)
    flux_args = (heat_flux, h_conv)
    material_args = (thermal_conductivity, specific_heat, density)

    args = (temperature_args, spacial_args, material_args, flux_args)

    results = explicit_method(max_time_steps, args)

    print(np.mean(results))