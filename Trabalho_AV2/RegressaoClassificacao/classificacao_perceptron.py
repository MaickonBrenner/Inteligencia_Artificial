import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, X, y, taxa_aprendizado=0.01, max_epocas=100, plot=False):
        self.p, self.N = X.shape
        self.X = np.vstack((-np.ones((1, self.N)), X))  # Adiciona bias -1
        self.y = y
        self.taxa = taxa_aprendizado
        self.max_epocas = max_epocas
        self.plot = plot
        self.w = np.random.rand(self.p + 1, 1)

    def func_ativacao(self, u):
        return np.where(u >= 0, 1, -1)

    def treino(self):
        historico_erros = []

        for epoca in range(self.max_epocas):
            erros = 0
            for k in range(self.N):
                x_k = self.X[:, k].reshape(-1, 1)
                d_k = self.y[k]
                u_k = (self.w.T @ x_k)[0, 0]  # Corrigido para evitar warning
                y_k = self.func_ativacao(u_k)
                if y_k != d_k:
                    self.w += self.taxa * (d_k - y_k) * x_k
                    erros += 1
            historico_erros.append(erros)
            if erros == 0:
                break

        if self.plot:
            plt.figure()
            plt.plot(historico_erros)
            plt.title("Curva de Aprendizado - Perceptron")
            plt.xlabel("Épocas")
            plt.ylabel("Nº de Erros")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def prever(self, X_teste):
        _, N = X_teste.shape
        X_teste = np.vstack((-np.ones((1, N)), X_teste))
        u = self.w.T @ X_teste
        return self.func_ativacao(u).flatten()

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
    dados = np.loadtxt("spiral3d.csv", delimiter=",")

    X = dados[:, :3]
    y = dados[:, 3]

    # Normalização Min-Max para [-1, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 2 * (X - X_min) / (X_max - X_min) - 1

    # Gráfico de dispersão 3D
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    cores = ['red' if c == 1 else 'blue' for c in y]
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=cores, s=10, alpha=0.6)
    ax.set_title("Dispersão 3D - spiral3d.csv")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    plt.tight_layout()
    plt.show()

    # Validação Monte Carlo
    R = 250
    taxa_aprendizado = 0.01
    max_epocas = 100

    acuracias, sensibilidades, especificidades = [], [], []

    for i in range(R):
        indices = np.random.permutation(X.shape[0])
        n_treino = int(0.8 * X.shape[0])
        idx_treino, idx_teste = indices[:n_treino], indices[n_treino:]

        X_treino = X[idx_treino].T
        y_treino = y[idx_treino]
        X_teste = X[idx_teste].T
        y_teste = y[idx_teste]

        modelo = Perceptron(X_treino, y_treino, taxa_aprendizado, max_epocas, plot=(i == 0))
        modelo.treino()
        y_pred = modelo.prever(X_teste)

        VP, VN, FP, FN = matriz_confusao(y_teste, y_pred)
        acc, sens, esp = metricas(VP, VN, FP, FN)

        acuracias.append(acc)
        sensibilidades.append(sens)
        especificidades.append(esp)

    # Resultados
    print("\nResultados - Perceptron Simples (Classificação - 250 rodadas):")
    for nome, lista in zip(["Acurácia", "Sensibilidade", "Especificidade"],
                           [acuracias, sensibilidades, especificidades]):
        print(f"\n{nome}:")
        print(f"  Média:  {np.mean(lista):.4f}")
        print(f"  Desvio: {np.std(lista):.4f}")
        print(f"  Máximo: {np.max(lista):.4f}")
        print(f"  Mínimo: {np.min(lista):.4f}")

if __name__ == "__main__":
    main()
