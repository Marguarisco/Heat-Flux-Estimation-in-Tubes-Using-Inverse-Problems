import numpy as np
import time

def dif_malhas(T_antes, T_novo):

    Ntheta_antes, Nr_antes = T_antes.shape # Dimensões da lista de temperaturas antes
    Ntheta_novo, Nr_novo = T_novo.shape # Dimensões da lista de temperaturas depois

    print(Nr_antes, Ntheta_antes)
    print(Nr_novo, Ntheta_novo)
    
    T_novo_interpol = np.zeros_like(T_antes)
    
    for i in range(Nr_antes):

        for j in range(Ntheta_antes):

            R_novo_index = int(i * (Nr_novo / Nr_antes))
            Theta_novo_index = int(j * (Ntheta_novo / Ntheta_antes))

            T_novo_interpol[j, i] = T_novo[Theta_novo_index, R_novo_index]
    
    # Calcular a diferença entre as duas soluções
    Erro = np.linalg.norm(T_antes - T_novo_interpol) / np.linalg.norm(T_antes)
    
    return Erro

# Função para convergência mantendo Ntheta fixo e variando radial_size
def convergir_r(method, Ntheta_inicial, Nr_inicial, Tolerancia, simulation_time, args):
    Convergiu = False
    Refino_r = 1
    T_antes = None

    method(simulation_time, args)
    
    while not Convergiu:
        Nr = Nr_inicial * Refino_r
        Ntheta = Ntheta_inicial  # Ntheta fixo

        args = physical_parameters(Nr, Ntheta)
        
        time_start = time.time()
        T_novo, _ = method(simulation_time, args)
        time_total = time.time() - time_start
        
        print(f"{Nr} x {Ntheta} - {time_total} - {Nr * Ntheta}")

        if T_antes is not None:
            Erro = dif_malhas(T_antes[-1], T_novo[-1])
            print(f"Erro em r (Ntheta fixo): {Erro}")
            
            if Erro < Tolerancia:
                print("Convergência em r alcançada")
                Convergiu = True

                return Nr_inicial * (Refino_r-1)
        
        Refino_r += 1
        T_antes = T_novo.copy()


# Função para convergência mantendo Nr fixo e variando Ntheta
def convergir_theta(method, Ntheta_inicial, Nr_inicial, Tolerancia, simulation_time, args):
    Convergiu = False
    Refino_theta = 1
    T_antes = None

    method(simulation_time, args)
    
    while not Convergiu:
        Nr = Nr_inicial  # Nr fixo
        Ntheta = Ntheta_inicial * Refino_theta

        args = physical_parameters(Nr, Ntheta)
        
        time_start = time.time()
        T_novo, _ = method(simulation_time, args)
        time_end = time.time()

        print(f"{Nr} x {Ntheta} - {time_end - time_start} - {Nr * Ntheta}")
        
        if T_antes is not None:
            Erro = dif_malhas(T_antes[-1], T_novo[-1])
            print(f"Erro em theta (Nr fixo): {Erro}")
            
            if Erro < Tolerancia:
                print("Convergência em theta alcançada")
                Convergiu = True

                return Ntheta_inicial * (Refino_theta-1)
        
        Refino_theta += 1
        T_antes = T_novo.copy()


if __name__ == '__main__':
    from physical_parameters import physical_parameters
    from Implicit_Method import implicit_method
    from Explicit_Method import explicit_method
    from ADI_Method import adi_method

    simulation_time = 1000000
    radial_size = 3
    angular_size = 4
    Tolerancia = 1e-5

    method = adi_method

    args = physical_parameters(radial_size, angular_size)

    # Primeira fase: convergência em r com Ntheta fixo
    #Nr_convergido = convergir_r(method, angular_size, radial_size, Tolerancia, simulation_time, args)
    Nr_convergido = 9

    # Segunda fase: convergência em theta com Nr fixo (usando a malha convergida em r)
    Ntheta_convergido = convergir_theta(method, angular_size, Nr_convergido, Tolerancia, simulation_time, args)

    print(f"Convergência completa em r sendo {Nr_convergido} e theta sendo {Ntheta_convergido}")