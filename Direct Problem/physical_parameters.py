import numpy as np

def physical_parameters(radial_size, angular_size):
    r_inner, r_outer = 0.1, 0.15

    dr = (r_outer - r_inner) / (radial_size - 1)  
    dtheta = 2 * np.pi / angular_size  

    radial_space = np.linspace(r_inner, r_outer, radial_size)
    angular_space = np.linspace(-np.pi, np.pi, angular_size, endpoint=False)
    

    T_inner, T_outer, T_tube = 300.0, 300.0, 300.0  #in Kelvin

    h_conv = 25.0
    heat_flux = ((-2000.0) * (angular_space / np.pi) ** 2) + 2000.0  
    #heat_flux = np.ones(angular_size) * 1000

    thermal_conductivity = 201.0  
    specific_heat = 900.0  
    density = 2700.0 

    spacial_args = (dr,  radial_size, radial_space, dtheta, angular_size, angular_space)
    temperature_args = (T_inner, T_outer, T_tube)
    flux_args = (heat_flux, h_conv)
    material_args = (thermal_conductivity, specific_heat, density)

    physical_args = (temperature_args, spacial_args, material_args, flux_args)

    return physical_args