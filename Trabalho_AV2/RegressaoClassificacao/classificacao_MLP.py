import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MLPBinaria:
    def __init__(self, n_input, n_hidden, taxa_aprendizado=0.01, max_epocas=200, precisao=1e-5):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.taxa = taxa_aprendizado
        self.max_epocas = max_epocas
        self.precisao = precisao
        self.losses = []
        self.w1 = np.random.randn(n_input + 1, n_hidden)
        self.w2 = np.random.randn(n_hidden + 1, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def fit(self, X, y):
        N = X.shape[0]
        erro_ant = np.inf
        X_bias = np.hstack((np.ones((N, 1)), X))
        for epoca in range(self.max_epocas):
            z1 = X_bias @ self.w1
            a1 = self.sigmoid(z1)
            a1_bias = np.hstack((np.ones((N, 1)), a1))
            z2 = a1_bias @ self.w2
            y_pred = self.sigmoid(z2)
            erro = y_pred - y
            mse = np.mean((erro) ** 2)
            self.losses.append(mse)
            if abs(erro_ant - mse) < self.precisao:
                break
            d2 = erro * self.sigmoid_deriv(z2)
            d1 = (d2 @ self.w2[1:].T) * self.sigmoid_deriv(z1)
            self.w2 -= self.taxa * a1_bias.T @ d2
            self.w1 -= self.taxa * X_bias.T @ d1
            erro_ant = mse

    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        a1 = self.sigmoid(X_bias @ self.w1)
        a1_bias = np.hstack((np.ones((N, 1)), a1))
        a2 = self.sigmoid(a1_bias @ self.w2)
        return np.where(a2 >= 0.5, 1, -1).flatten()

def matriz_confusao(y_real, y_pred):
    VP = np.sum((y_real == 1) & (y_pred == 1))
    VN = np.sum((y_real == -1) & (y_pred == -1))
    FP = np.sum((y_real == -1) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == -1))
    return VP, VN, FP, FN

def metricas(VP, VN, FP, FN):
    total = VP + VN + FP + FN
    acuracia = (VP + VN) / total if total else 0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0
    return acuracia, sensibilidade, especificidade

def main():
    np.random.seed(42)
    dados = np.loadtxt("Spiral3d.csv", delimiter=",")
    X = dados[:, :3]
    y = dados[:, 3].reshape(-1, 1)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 2 * (X - X_min) / (X_max - X_min) - 1

    n_hidden = 10
    R = 250
    taxa = 0.01
    max_epocas = 200
    precisao = 1e-5

    accs, sens_list, esp_list = [], [], []
    melhor_loss, pior_loss = [], []
    melhor_acc, pior_acc = 0, 1
    y_melhor_real, y_melhor_pred = None, None
    y_pior_real, y_pior_pred = None, None

    for i in range(R):
        indices = np.random.permutation(X.shape[0])
        n_treino = int(0.8 * X.shape[0])
        idx_treino, idx_teste = indices[:n_treino], indices[n_treino:]

        X_train = X[idx_treino]
        y_train = y[idx_treino]
        X_test = X[idx_teste]
        y_test = y[idx_teste]

        mlp = MLPBinaria(n_input=3, n_hidden=n_hidden, taxa_aprendizado=taxa,
                         max_epocas=max_epocas, precisao=precisao)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        VP, VN, FP, FN = matriz_confusao(y_test.flatten(), y_pred)
        acc, sens, esp = metricas(VP, VN, FP, FN)

        accs.append(acc)
        sens_list.append(sens)
        esp_list.append(esp)

        if acc > melhor_acc:
            melhor_acc = acc
            melhor_loss = mlp.losses.copy()
            y_melhor_real = y_test.copy()
            y_melhor_pred = y_pred.copy()

        if acc < pior_acc:
            pior_acc = acc
            pior_loss = mlp.losses.copy()
            y_pior_real = y_test.copy()
            y_pior_pred = y_pred.copy()

        if i % 25 == 0:
            print(f"Rodada {i}/{R}")

    print("\nResultados - MLP (Classificação Binária - 250 rodadas):")
    for nome, lista in zip(["Acurácia", "Sensibilidade", "Especificidade"],
                           [accs, sens_list, esp_list]):
        print(f"\n{nome}:")
        print(f"  Média:  {np.mean(lista):.4f}")
        print(f"  Desvio: {np.std(lista):.4f}")
        print(f"  Máximo: {np.max(lista):.4f}")
        print(f"  Mínimo: {np.min(lista):.4f}")

    # Curvas de aprendizado do melhor e pior caso
    if melhor_loss and pior_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(melhor_loss, label="Melhor caso")
        plt.plot(pior_loss, label="Pior caso")
        plt.title("Curvas de Aprendizado - MLP Binário")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio (MSE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Matrizes de confusão (melhor e pior)
    for tipo, y_real, y_pred in zip(["Melhor", "Pior"],
                                    [y_melhor_real, y_pior_real],
                                    [y_melhor_pred, y_pior_pred]):
        VP, VN, FP, FN = matriz_confusao(y_real.flatten(), y_pred)
        matriz = np.array([[VP, FN],
                           [FP, VN]])
        plt.figure()
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {tipo} Caso")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()