import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, X_treino, Y_treino, num_max_epoca, taxa_aprendizagem, precisao, plot=True):
        self.p, self.N = X_treino.shape
        self.X_treino = np.vstack((-np.ones((1, self.N)), X_treino))  # adiciona bias -1
        self.Y_treino = Y_treino
        self.taxa_aprendizagem = taxa_aprendizagem
        self.precisao = precisao
        self.num_max_epoca = num_max_epoca
        self.w = np.random.rand(self.p + 1, 1)
        self.plot = plot

    def treino(self):
        epoca = 0
        eqm_ant = float("inf")
        hist_eqm = []

        while epoca < self.num_max_epoca:
            eqm = self.__EQM()
            hist_eqm.append(eqm)

            if abs(eqm_ant - eqm) < self.precisao:
                break

            for k in range(self.N):
                x_k = self.X_treino[:, k].reshape(self.p + 1, 1)
                d_k = self.Y_treino[k]
                u_k = (self.w.T @ x_k)[0, 0]
                e_k = d_k - u_k
                self.w += self.taxa_aprendizagem * e_k * x_k

            eqm_ant = eqm
            epoca += 1

        if self.plot:
            plt.figure()
            plt.plot(hist_eqm)
            plt.xlabel("Épocas")
            plt.ylabel("Erro Quadrático Médio (EQM)")
            plt.title("Curva de Aprendizado - ADALINE")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def prever(self, X):
        _, N = X.shape
        X_bias = np.vstack((-np.ones((1, N)), X))
        return (self.w.T @ X_bias).flatten()

    def __EQM(self):
        erro_total = 0
        for k in range(self.N):
            x_k = self.X_treino[:, k].reshape(self.p + 1, 1)
            d_k = self.Y_treino[k]
            u_k = (self.w.T @ x_k)[0, 0]
            erro_total += (d_k - u_k) ** 2
        return erro_total / (2 * self.N)


def main():
    # Carregar e limpar dados
    dados = np.loadtxt("aerogerador.dat")
    dados = dados[~np.all(dados == 0, axis=1)]  # remove linhas com 0, 0

    X = dados[:, 0].reshape(-1, 1)
    y = dados[:, 1].reshape(-1, 1)

    # Visualização inicial
    plt.figure()
    plt.scatter(X, y, s=10, alpha=0.5)
    plt.xlabel("Velocidade do Vento")
    plt.ylabel("Potência Gerada")
    plt.title("Dispersão dos Dados do Aerogerador")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Normalização Z-score
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()

    X_norm_T = X_norm.T  # formato p x N
    y_norm_flat = y_norm.flatten()

    # Parâmetros
    R = 250
    taxa_aprendizagem = 0.0001
    num_max_epoca = 200
    precisao = 1e-5
    mses = []

    # Validação Monte Carlo
    for i in range(R):
        indices = np.random.permutation(X_norm.shape[0])
        n_teste = int(0.2 * X_norm.shape[0])
        idx_teste = indices[:n_teste]
        idx_treino = indices[n_teste:]

        X_treino = X_norm[idx_treino].T
        y_treino = y_norm[idx_treino].flatten()
        X_teste = X_norm[idx_teste].T
        y_teste = y_norm[idx_teste].flatten()

        modelo = Adaline(X_treino, y_treino, num_max_epoca, taxa_aprendizagem, precisao, plot=(i == 0))
        modelo.treino()
        y_pred = modelo.prever(X_teste)
        mse = np.mean((y_teste - y_pred) ** 2)
        mses.append(mse)

    # Resultados
    print("\nResultados do ADALINE (Regressão - 250 rodadas):")
    print(f"Média do MSE:          {np.mean(mses):.6f}")
    print(f"Desvio Padrão do MSE: {np.std(mses):.6f}")
    print(f"Maior MSE:             {np.max(mses):.6f}")
    print(f"Menor MSE:             {np.min(mses):.6f}")

if __name__ == "__main__":
    main()
