import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, arquivo, tamanho_populacao, geracoes, taxa_mutacao, tamanho_torneio):
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.tamanho_torneio = tamanho_torneio

        self.pontos = np.random.randint(30,60)
        data = np.loadtxt(arquivo, delimiter=',')
        
        if self.pontos < data.shape[0]:
            indices = np.random.choice(data.shape[0], self.pontos, replace=False)
            self.coordenadas = data[indices]
        else:
            self.coordenadas = data

        self.populacao = [np.random.permutation(self.pontos) for i in range(self.tamanho_populacao)]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')  # Gráfico em 3D
        self.linhas = []
        self.ax.scatter(self.coordenadas[:,0], self.coordenadas[:,1], self.coordenadas[:,2], c='r', marker='o')
        self.plot_rota()

    def distancia_euclidiana(self, pontoA, pontoB):
        return np.sqrt(np.sum((pontoA - pontoB)**2))
    
    def custo_rota(self, rota):
        return sum(self.distancia_euclidiana(self.coordenadas[rota[i]], self.coordenadas[rota[i + 1]])
                for i in range(len(rota)-1))
    
    def torneio(self):
        competidores = np.random.choice(self.tamanho_populacao, self.tamanho_torneio, replace=False)
        melhor_idx = min(competidores, key=lambda i: self.custo_rota(self.populacao[i]))
        return self.populacao[melhor_idx]

    def recombinacao(self, idv1, idv2):  
        tamanho = len(idv1)
        inicio, fim = sorted(np.random.choice(tamanho, 2, replace=False))
        prole1 = -np.ones(tamanho, dtype=int)
        prole2 = -np.ones(tamanho, dtype=int)
        prole1[inicio:fim] = idv1[inicio:fim]
        prole2[inicio:fim] = idv2[inicio:fim]

        def prole_completa(prole, idv):
            restante = [gene for gene in idv if gene not in prole]
            idx = np.where(prole == -1)[0]
            prole[idx] = restante
            return prole
        return prole_completa(prole1, idv2), prole_completa(prole2, idv1)
    
    def mutacao(self, individual):
        if np.random.rand() < self.taxa_mutacao:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def evolucao(self):
        for geracao in range(self.geracoes):
            nova_populacao = []
            for _ in range(self.tamanho_populacao // 2):
                idv1 = self.torneio()
                idv2 = self.torneio()
                prole1, prole2 = self.recombinacao(idv1, idv2)
                prole1 = self.mutacao(prole1)
                prole2 = self.mutacao(prole2)
                nova_populacao.extend([prole1, prole2])
            self.populacao = nova_populacao

        self.plot_rota()
        plt.title(f"Geração {geracao}")
        plt.pause(0.1)

    def melhor_rota(self):
        return min(self.populacao, key=self.custo_rota)
    
    def plot_rota(self, cor='k'):
        self.ax.clear()
        self.ax.scatter(self.coordenadas[:,0], self.coordenadas[:,1], self.coordenadas[:,2], c='r', marker='o')

        melhor_caminho = self.melhor_rota()
        ponto_origem = self.coordenadas[melhor_caminho[0]]
        self.ax.scatter(ponto_origem[0], ponto_origem[1], ponto_origem[2], c="gold", s=100, marker="*", edgecolors="k", label="Origem")
        for i in range(len(melhor_caminho)):
            p1 = self.coordenadas[melhor_caminho[i]]
            p2 = self.coordenadas[melhor_caminho[(i+1) % len(melhor_caminho)]]

            if i == 0:
                l = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='yellow', label="Início")  # Início
            elif i == len(melhor_caminho) - 1:
                l = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='cyan', label="Fim")  # Fim
            else:
                l = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=cor)  # Intermediário
            self.linhas.append(l)

        plt.title("Melhor Caminho Encontrado")
        plt.pause(0.1)

    def plot_rota_final(self):
        self.plot_rota(cor='g')
        plt.title("Rota Final Subótima")
        plt.legend()
        plt.show()
