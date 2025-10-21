Este repositório contém os códigos-fonte e materiais desenvolvidos para o **Projeto de Graduação em Engenharia Mecânica**, apresentado à **Escola Politécnica da Universidade Federal do Rio de Janeiro (UFRJ)**.

O objetivo principal deste trabalho é desenvolver e aplicar métodos de solução de problemas inversos para estimar o fluxo de calor em uma seção transversal bidimensional de um tubo. A metodologia aborda tanto cenários de fluxo de calor permanente quanto transiente.

Os algoritmos foram implementados em **Python (versão 3.10.1)**. As principais bibliotecas utilizadas para a execução dos códigos são:

  - **NumPy**: Para manipulação eficiente de matrizes e cálculos numéricos.
  
  - **SciPy**: Para funções matemáticas otimizadas.
  
  - **Numba**: Para otimização e aceleração dos loops computacionais no problema direto.
  
  - **Multiprocessing**: Para paralelizar o cálculo do gradiente no problema inverso, reduzindo o tempo de execução.
