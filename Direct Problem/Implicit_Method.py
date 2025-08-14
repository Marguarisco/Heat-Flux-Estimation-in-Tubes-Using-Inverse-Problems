import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True)
def implicit_method(simulation_time, args):

    temperature_args, spacial_args, material_args, flux_args = args

    T_inner, T_outer, T_tube = temperature_args
    dr, radial_size, radial_space, dtheta, angular_size, angular_space = spacial_args
    thermal_conductivity, specific_heat, density = material_args
    heat_flux, h_conv = flux_args

    thermal_diffusivity = thermal_conductivity / (specific_heat * density)  

    dt = 0.1

    current_temp = np.ones((angular_size, radial_size)) * T_tube

    total_points = angular_size * radial_size
    coefficent_vector = np.identity(total_points, dtype=np.float64)
    rhs_vector = np.zeros(total_points, dtype=np.float64) # Right-hand side vector

    gamma_r  = thermal_diffusivity * dt / (2 * dr * radial_space)
    gamma_rr = thermal_diffusivity * dt / (dr ** 2)
    gamma_tt = thermal_diffusivity * dt / ((dtheta ** 2) * (radial_space ** 2))

    beta_0 = (2 * dr * heat_flux) / thermal_conductivity
    gamma_0 = (- gamma_r[0] + gamma_rr) * beta_0

    beta_I = 2 * dr * h_conv / thermal_conductivity
    gamma_I = (gamma_r[-1] + gamma_rr) * beta_I 

    internal_i = [i for i in range(total_points) if i % radial_size == 0]
    external_i = [i for i in range(total_points) if i % radial_size == radial_size - 1]
    middle_i = [i for i in range(total_points) if i not in internal_i and i not in external_i]

    for i in middle_i:
        Raio = i % (radial_size)
        gamma_r_raio = gamma_r[Raio]
        gamma_tt_raio = gamma_tt[Raio]

        coefficent_vector[i,i - 1] = - gamma_rr + gamma_r_raio
        coefficent_vector[i,i] = 1 + (2 * gamma_rr) + (2 * gamma_tt_raio)
        coefficent_vector[i,i + 1] = - gamma_rr - gamma_r_raio

        coefficent_vector[i,i - radial_size] = - gamma_tt_raio
        
        position_theta = i + radial_size
        if (i + radial_size) > max(middle_i):
            position_theta = i % radial_size
    
        coefficent_vector[i,position_theta] = - gamma_tt_raio

    for i in internal_i:
        gamma_tt_raio = gamma_tt[0]
        coefficent_vector[i,i] = 1 + 2 * gamma_rr + 2 * gamma_tt_raio
        coefficent_vector[i,i + 1] = - 2 * gamma_rr

        coefficent_vector[i,i - radial_size] = - gamma_tt_raio

        position_theta = i + radial_size
        if (i + radial_size) > max(internal_i):
            position_theta = i % radial_size
    
        coefficent_vector[i,position_theta] = - gamma_tt_raio


    for i in external_i:
        gamma_tt_raio = gamma_tt[-1] 

        coefficent_vector[i,i] = 1 + 2 * gamma_rr + 2 * gamma_tt_raio + gamma_I
        coefficent_vector[i,i - 1] = - 2 * gamma_rr

        coefficent_vector[i,i - radial_size] = - gamma_tt_raio

        position_theta = i + radial_size
        if (i + radial_size) > max(external_i):
            position_theta = i % radial_size                        
    
        coefficent_vector[i,position_theta] = - gamma_tt_raio

    T_ext_history = np.zeros((simulation_time, angular_size, radial_size), dtype=np.float64)
    T_ext_history[0, :, :] = current_temp

    time_step = 1
    while time_step < simulation_time:
        

        for _ in range(1):
            current_temp_flat = current_temp.flatten()

            for i in middle_i:
                rhs_vector[i] = current_temp_flat[i]

            for i in internal_i:
                rhs_vector[i] = current_temp_flat[i] + gamma_0[i//radial_size]
            
            for i in external_i:
                rhs_vector[i] = current_temp_flat[i] + (T_outer * gamma_I)
            
                
            current_temp = np.linalg.solve(coefficent_vector, rhs_vector).reshape((angular_size, radial_size))  
        
        T_ext_history[time_step, :, :] = current_temp

        time_step += 1

    return T_ext_history, dt

if __name__ == '__main__':
    max_time_steps = 100

    radial_size = 25
    angular_size = 3

    r_inner, r_outer = 0.1, 0.15

    dr = (r_outer - r_inner) / (radial_size - 1)  
    dtheta = 2 * np.pi / angular_size  

    radial_space = np.linspace(r_inner, r_outer, radial_size)
    angular_space = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)
    

    T_inner, T_outer, T_tube = 300.0, 300.0, 300.0  

    h_conv = 25.0
    heat_flux = ((-2000.0) * (angular_space / np.pi) ** 2) + 2000.0  
    heat_flux = np.ones(angular_size) * 1000

    thermal_conductivity = 201.0  
    specific_heat = 900.0  
    density = 2700.0 

    spacial_args = (dr,  radial_size, radial_space, dtheta, angular_size, angular_space)
    temperature_args = (T_inner, T_outer, T_tube)
    flux_args = (heat_flux, h_conv)
    material_args = (thermal_conductivity, specific_heat, density)

    args = (max_time_steps, temperature_args, spacial_args, material_args, flux_args)

    results = implicit_method(args)
    

    print(np.mean(results))