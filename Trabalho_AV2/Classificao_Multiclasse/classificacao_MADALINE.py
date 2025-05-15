import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def one_hot_encode(rotulos):
    mapeamento = {
        0: [1, -1, -1],  # Normal
        1: [-1, 1, -1],  # Hérnia de Disco
        2: [-1, -1, 1]   # Espondilolistese
    }
    return np.array([mapeamento[rotulo] for rotulo in rotulos])

def calcular_metricas(matriz_confusao):
    total_amostras = np.sum(matriz_confusao)
    acuracia = np.trace(matriz_confusao) / total_amostras if total_amostras > 0 else 0
    sensibilidade = []
    especificidade = []

    for i in range(matriz_confusao.shape[0]):
        VP = matriz_confusao[i, i]
        FN = np.sum(matriz_confusao[i, :]) - VP
        FP = np.sum(matriz_confusao[:, i]) - VP
        VN = total_amostras - (VP + FP + FN)

        sensibilidade.append(VP / (VP + FN) if (VP + FN) > 0 else 0)
        especificidade.append(VN / (VN + FP) if (VN + FP) > 0 else 0)

    return acuracia, sensibilidade, especificidade

def matriz_confusao(y_real, y_pred):
    reais = np.argmax(y_real, axis=1)
    num_classes = len(np.unique(reais))
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for verdadeiro, previsto in zip(reais, y_pred):
        matriz[verdadeiro, previsto] += 1
    return matriz

def curva_aprendizagem(hist_eqm):
    plt.figure()
    plt.plot(hist_eqm)
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.title("Curva de Aprendizagem")
    plt.show()

class Madaline:
    def __init__(self, n_entradas, n_ocultos, taxa_aprendizagem):
        self.n_entradas = n_entradas
        self.n_ocultos = n_ocultos
        self.taxa_aprendizagem = taxa_aprendizagem

        self.W_oculta = np.random.randn(n_ocultos, n_entradas + 1)
        self.W_saida = np.random.randn(n_ocultos + 1)

    def _ativacao_oculta(self, x):
        return np.tanh(x)

    def _derivada_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def treino(self, X, Y, max_epocas=500, precisao=1e-3, plot=False):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        erro_anterior = np.inf
        hist_eqm = []

        for epoca in range(max_epocas):
            eqm_total = 0
            for i in range(N):
                x_i = X_bias[i]
                y_i = Y[i]

                z_oculta = self._ativacao_oculta(self.W_oculta @ x_i)
                z_oculta_bias = np.hstack(([1], z_oculta))
                y_hat = self.W_saida @ z_oculta_bias

                erro = y_i - y_hat
                eqm_total += erro ** 2

                grad_saida = erro * z_oculta_bias
                grad_oculta = erro * self.W_saida[1:] * self._derivada_tanh(self.W_oculta @ x_i)
                self.W_saida += self.taxa_aprendizagem * grad_saida
                self.W_oculta += self.taxa_aprendizagem * grad_oculta[:, np.newaxis] * x_i

            eqm_medio = eqm_total / (2 * N)
            hist_eqm.append(eqm_medio)

            if abs(erro_anterior - eqm_medio) < precisao:
                break
            erro_anterior = eqm_medio

        if plot:
            curva_aprendizagem(hist_eqm)

    def predizer(self, X):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        saidas = []
        for i in range(N):
            x_i = X_bias[i]
            z_oculta = self._ativacao_oculta(self.W_oculta @ x_i)
            z_oculta_bias = np.hstack(([1], z_oculta))
            y_hat = self.W_saida @ z_oculta_bias
            saidas.append(y_hat)
        return np.where(np.array(saidas) >= 0, 1, -1)

def main():
    dados = np.loadtxt("coluna_vertebral.csv", delimiter=",", dtype=str, encoding="utf-8")
    X = dados[:, :-1].astype(float)
    Y_dados = dados[:, -1]
    rotulos = {"NO": 0, "DH": 1, "SL": 2}
    Y = np.array([rotulos[y] for y in Y_dados])
    Y_codificado = one_hot_encode(Y)

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_normalizado = (X - X_min) / (X_max - X_min)

    rodadas = 100
    particionamento = 0.8
    max_epoca = 500
    taxa_aprendizagem = 0.01
    precisao = 0.001
    n_ocultos = 3

    acuracias, sensibilidades, especificidades = [], [], []
    matrizes_confusao = []
    n_amostras = X.shape[0]
    n_treino = int(particionamento * n_amostras)

    for i in range(rodadas):
        indices = np.random.permutation(n_amostras)
        indices_treino, indices_teste = indices[:n_treino], indices[n_treino:]

        X_treino, Y_treino = X_normalizado[indices_treino], Y_codificado[indices_treino]
        X_teste, Y_teste = X_normalizado[indices_teste], Y_codificado[indices_teste]

        modelos = []
        for classe in range(3):
            Y_bin = 2 * (Y_treino[:, classe] == 1) - 1
            modelo = Madaline(X.shape[1], n_ocultos, taxa_aprendizagem)
            modelo.treino(X_treino, Y_bin, max_epoca, precisao, plot=(i == 0 and classe == 0))
            modelos.append(modelo)

        saidas = np.column_stack([modelo.predizer(X_teste) for modelo in modelos])
        Y_pred = np.argmax(saidas, axis=1)

        matriz = matriz_confusao(Y_teste, Y_pred)
        acuracia, sens, espec = calcular_metricas(matriz)

        acuracias.append(acuracia)
        sensibilidades.append(sens)
        especificidades.append(espec)
        matrizes_confusao.append(matriz)

    sensibilidades = np.array(sensibilidades)
    especificidades = np.array(especificidades)

    print(f"Média Acurácia: {np.mean(acuracias):.4f} | Desvio: {np.std(acuracias):.4f}")
    print(f"Média Sensibilidade: {np.mean(sensibilidades, axis=0)}")
    print(f"Média Especificidade: {np.mean(especificidades, axis=0)}")

    media = np.mean(acuracias)
    desvio = np.std(acuracias)
    maximo = np.max(acuracias)
    minimo = np.min(acuracias)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=acuracias, ax=axes[0])
    axes[0].set_title("Boxplot da Acurácia")
    axes[0].annotate(f"Média: {media:.4f}\nDesvio: {desvio:.4f}\nMáx: {maximo:.4f}\nMín: {minimo:.4f}",
                     xy=(0.7, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="w"))
    sns.violinplot(y=acuracias, ax=axes[1])
    axes[1].set_title("Violin Plot da Acurácia")
    axes[1].annotate(f"Média: {media:.4f}\nDesvio: {desvio:.4f}\nMáx: {maximo:.4f}\nMín: {minimo:.4f}",
                     xy=(0.7, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="w"))
    plt.tight_layout()
    plt.show()

    idx_max, idx_min = np.argmax(acuracias), np.argmin(acuracias)
    for idx, titulo, cor in [(idx_max, "Maior Acurácia", "Blues"), (idx_min, "Menor Acurácia", "Reds")]:
        matriz = matrizes_confusao[idx]
        plt.figure()
        sns.heatmap(matriz, annot=True, cmap=cor, fmt="d")
        plt.title(f"{titulo} ({acuracias[idx]:.4f})")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()

if __name__ == "__main__":
    main()