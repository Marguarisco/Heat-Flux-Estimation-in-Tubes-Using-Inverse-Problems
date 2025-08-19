from physical_parameters import physical_parameters
import numpy as np
import time

def main():
    simulation_time = 6000

    radial_size = 25
    angular_size = 3

    args = physical_parameters(radial_size, angular_size)

    start_time = time.time()
    start_time_process = time.process_time()
    if method == 'I':
        from Implicit_Method import implicit_method
        print("Iniciando a simulação...")
        results, dt = implicit_method(simulation_time, args)

        method_arquive = "resultados_implicit_completo"
        print("Simulação concluída.")

    elif method == 'E':
        from Explicit_Method import explicit_method
        print("Iniciando a simulação...")
        results, dt = explicit_method(simulation_time, args)

        method_arquive = "resultados_explicit_completo"
        print("Simulação concluída.")

    else:
        from ADI_Method import adi_method
        print("Iniciando a simulação...")
        results, dt = adi_method(simulation_time, args)

        method_arquive = "resultados_adi_completo"
        print("Simulação concluída.")
    
    final_time = time.time()
    final_time_process = time.process_time()

    running_time = final_time - start_time
    running_time_process = final_time_process - start_time_process

    path = "C:/Users/marce/Desktop/TCC/Direct Problem/Results/"

    print(f"Salvando resultados com a forma: {results.shape}")

    temperature_args, spacial_args, material_args, flux_args = args
    dr, radial_size_val, radial_space, dtheta, angular_size_val, angular_space = spacial_args


    np.savez_compressed(
        path + method_arquive + f'_{radial_size}_{angular_size}_{simulation_time}_dt.npz', 
        temperaturas=results,
        running_time = running_time,
        running_time_process = running_time_process,
        dt = dt,
        
        # Parâmetros espaciais
        radial_space=radial_space,
        angular_space=angular_space,
        dr=dr,
        dtheta=dtheta,

        # Condições de contorno e fluxo
        heat_flux=flux_args[0],
        h_conv=flux_args[1],
        T_initial=temperature_args[2], # T_tube

        # Propriedades do material (extraídas de 'material_args')
        thermal_conductivity=material_args[0],
        specific_heat=material_args[1],
        density=material_args[2]
    )

    print("Arquivo salvo com sucesso!")


method = 'E'
results = main()

    

    

    

    