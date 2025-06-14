import numpy as np
import matplotlib.pyplot as plt
from algoritmo_genetico import GeneticAlgorithm

# data = np.loadtxt("CaixeiroGruposGA.csv",delimiter=',')

ga = GeneticAlgorithm("CaixeiroGruposGA.csv", 50, 100, 0.1, 5)
ga.evolucao()
ga.plot_rota_final()

# c1, c2, c3, c4 = 1,2,3,4

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[data[:,-1]==c1,0], data[data[:,-1]==c1,1], data[data[:,-1]==c1,2], label='Grupo 1', c='yellow', edgecolor='k')
# ax.scatter(data[data[:,-1]==c2,0], data[data[:,-1]==c2,1], data[data[:,-1]==c2,2], label='Grupo 2', c='teal', edgecolor='k')
# ax.scatter(data[data[:,-1]==c3,0], data[data[:,-1]==c3,1], data[data[:,-1]==c3,2], label='Grupo 3', c='red', edgecolor='k')
# ax.scatter(data[data[:,-1]==c4,0], data[data[:,-1]==c4,1], data[data[:,-1]==c4,2], label='Grupo 4', c='blue', edgecolor='k')
# plt.legend()
# plt.show()