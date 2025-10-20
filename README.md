Este repositório contém os códigos-fonte e materiais desenvolvidos para o **Projeto de Graduação em Engenharia Mecânica**, apresentado à **Escola Politécnica da Universidade Federal do Rio de Janeiro (UFRJ)**.

O objetivo principal deste trabalho é desenvolver e aplicar métodos de solução de problemas inversos para estimar o fluxo de calor em uma seção transversal bidimensional de um tubo. A metodologia aborda tanto cenários de fluxo de calor permanente quanto transiente.

Os algoritmos foram implementados em **Python (versão 3.10.1)**. As principais bibliotecas utilizadas para a execução dos códigos são:

  - **NumPy**: Para manipulação eficiente de matrizes e cálculos numéricos.
  
  - **SciPy**: Para funções matemáticas otimizadas.
  
  - **Matplotlib**: Para a geração de todos os gráficos e visualizações dos resultados.
  
  - **Numba**: Para otimização JIT (Just-In-Time) e aceleração dos loops computacionais no problema direto.
  
  - **Multiprocessing**: Para paralelizar o cálculo do gradiente no problema inverso, reduzindo o tempo de execução.
  
  - **h5py**: Para armazenamento e gerenciamento eficiente dos dados de simulação em formato binário (HDF5).
