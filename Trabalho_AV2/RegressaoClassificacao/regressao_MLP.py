import numpy as np
import matplotlib.pyplot as plt


class MLPRegressor:
    def __init__(self, n_input, n_hidden, n_output, taxa_aprendizagem, max_epocas, precisao, plot=False):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.taxa_aprendizagem = taxa_aprendizagem
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.plot = plot

        self.w1 = np.random.randn(n_hidden, n_input + 1) * 0.1
        self.w2 = np.random.randn(n_output, n_hidden + 1) * 0.1

    def ativacao(self, x):
        return np.tanh(x)

    def derivada_ativacao(self, x):
        return 1.0 - np.tanh(x) ** 2

    def treino(self, X, y):
        p, N = X.shape
        X_bias = np.vstack([-np.ones((1, N)), X])
        y = y.reshape(1, N)

        hist_eqm = []
        eqm_anterior = float("inf")

        for epoca in range(self.max_epocas):
            z_in = self.w1 @ X_bias
            z = self.ativacao(z_in)
            z_bias = np.vstack([-np.ones((1, N)), z])

            y_in = self.w2 @ z_bias
            y_hat = y_in

            erro = y - y_hat
            EQM = np.mean(erro ** 2) / 2
            hist_eqm.append(EQM)

            if abs(eqm_anterior - EQM) < self.precisao:
                break
            eqm_anterior = EQM

            delta2 = -erro
            delta1 = (self.w2[:, 1:].T @ delta2) * self.derivada_ativacao(z_in)

            self.w2 -= self.taxa_aprendizagem * (delta2 @ z_bias.T) / N
            self.w1 -= self.taxa_aprendizagem * (delta1 @ X_bias.T) / N

        if self.plot:
            plt.figure()
            plt.plot(hist_eqm)
            plt.xlabel("Épocas")
            plt.ylabel("EQM")
            plt.title("Curva de Aprendizado - MLP")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def prever(self, X):
        _, N = X.shape
        X_bias = np.vstack([-np.ones((1, N)), X])
        z = self.ativacao(self.w1 @ X_bias)
        z_bias = np.vstack([-np.ones((1, N)), z])
        y_hat = self.w2 @ z_bias
        return y_hat.flatten()


def main():
    dados = np.loadtxt("aerogerador.dat")
    dados = dados[~np.all(dados == 0, axis=1)]

    X = dados[:, 0].reshape(-1, 1)
    y = dados[:, 1].reshape(-1, 1)

    plt.figure()
    plt.scatter(X, y, s=10, alpha=0.5)
    plt.xlabel("Velocidade do Vento")
    plt.ylabel("Potência Gerada")
    plt.title("Dispersão dos Dados do Aerogerador")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Normalização Min-Max [-1, 1]
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

    X_norm_T = X_norm.T
    y_norm_flat = y_norm.flatten()

    # Parâmetros
    R = 250
    taxa_aprendizagem = 0.01
    num_max_epoca = 500
    precisao = 1e-5
    mses = []

    for i in range(R):
        indices = np.random.permutation(X_norm.shape[0])
        n_teste = int(0.2 * X_norm.shape[0])
        idx_teste = indices[:n_teste]
        idx_treino = indices[n_teste:]

        X_treino = X_norm[idx_treino].T
        y_treino = y_norm[idx_treino].flatten()
        X_teste = X_norm[idx_teste].T
        y_teste = y_norm[idx_teste].flatten()

        modelo = MLPRegressor(
            n_input=1,
            n_hidden=10,  #para testar under/overfitting
            n_output=1,
            taxa_aprendizagem=taxa_aprendizagem,
            max_epocas=num_max_epoca,
            precisao=precisao,
            plot=(i == 0)
        )
        modelo.treino(X_treino, y_treino)
        y_pred = modelo.prever(X_teste)
        mse = np.mean((y_teste - y_pred) ** 2)
        mses.append(mse)

    print("\nResultados do MLP (Regressão - 250 rodadas):")
    print(f"Média do MSE:          {np.mean(mses):.4f}")
    print(f"Desvio Padrão do MSE: {np.std(mses):.4f}")
    print(f"Maior MSE:             {np.max(mses):.4f}")
    print(f"Menor MSE:             {np.min(mses):.4f}")

if __name__ == "__main__":
    main()
