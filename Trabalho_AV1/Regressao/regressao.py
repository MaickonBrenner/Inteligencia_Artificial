import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("atividade_enzimatica.csv", delimiter=',')

# 1 - Temperatura, 2 - ph da Solução, 3 - Nivel de atividade enzimática

X = data[:,:2]
y = data[:,2:]

plt.figure(0)
ax = plt.subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],y[:,0],c='purple',edgecolor='k')
ax.set_xlabel("Temperatura")
ax.set_ylabel("Ph da Solução")
ax.set_zlabel("Nível de atividade enzimática")
ax.set_title("Análise dos Níveis de Atividade Enzimática")

# Simulações por Monte Carlo

# Quantidade de rodadas:
rodadas = 1

# Particionamento dos Dados (80% treino, 20% teste):
particionamento = 0.8

# Desempenhos:
modelo_media = []
modelo_mqo_s = []
modelo_mqo = []

plot_graphs = True
idx_plot = 1

plt.show()