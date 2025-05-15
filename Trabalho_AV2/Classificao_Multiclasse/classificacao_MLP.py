import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

def one_hot_encode(rotulos):
    mapeamento = {
        0: [1, -1, -1], # Normal
        1: [-1, 1, -1], # Hérnia de Disco
        2: [-1, -1, 1]  # Espondilolistese
    }
    return np.array([mapeamento[rotulo] for rotulo in rotulos])

def calcular_metricas(matriz):
    acuracia = np.trace(matriz) / np.sum(matriz)

    sensibilidades_classe = []
    especificidades_classe = []

    for i in range(matriz.shape[0]):  # Para cada classe
        VP = matriz[i, i]
        FP = np.sum(matriz[:, i]) - VP
        FN = np.sum(matriz[i, :]) - VP
        VN = np.sum(matriz) - (VP + FP + FN)

        sens = VP / (VP + FN) if (VP + FN) > 0 else 0
        espec = VN / (VN + FP) if (VN + FP) > 0 else 0

        sensibilidades_classe.append(sens)
        especificidades_classe.append(espec)

    sensibilidade = np.mean(sensibilidades_classe)
    especificidade = np.mean(especificidades_classe)

    return acuracia, sensibilidade, especificidade

def matriz_confusao(y_real, y_pred, num_classes=3):
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for r, p in zip(y_real, y_pred):
        matriz[r, p] += 1
    return matriz

def curva_aprendizagem(hist_eqm):
    plt.figure(2)
    plt.plot(hist_eqm)
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.title("Curva de Aprendizagem")

class MLP:
    def __init__(self, input, hidden, output, taxa_aprendizagem):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.taxa_aprendizagem = taxa_aprendizagem
        self.w1 = np.random.randn(self.input + 1, self.hidden)
        self.w2 = np.random.randn(self.hidden + 1, self.output)

    def treino(self, X, Y, max_epocas=1000, precisao=1e-3):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)) * 1, X))
        erro_anterior = np.inf 
        hist_eqm = []

        for epoca in range(max_epocas): 
            hist_eqm.append(erro_anterior)
            # Forward
            Z_in = X_bias@self.w1
            Z = sigmoid(Z_in)
            Z_bias = np.hstack((np.ones((N, 1)) * 1, Z))
            Y_in = Z_bias@self.w2
            Y_hat = sigmoid(Y_in)

            # Erro ou Loss
            erro = Y - Y_hat
            eqm = np.mean(erro**2)/2

            # Backpropagation
            d2 = erro * sigmoid_derivada(Y_hat)
            d1 = (d2@self.w2[1:].T)*sigmoid_derivada(Z)

            # Atualização dos pesos
            self.w2 += self.taxa_aprendizagem * Z_bias.T@d2
            self.w1 += self.taxa_aprendizagem * X_bias.T@d1

            if abs(erro_anterior - eqm) < precisao:
                break
            erro_anterior = eqm
            
        curva_aprendizagem(hist_eqm)

    def predizer(self, X):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)) * 1, X))
        Z = sigmoid(X_bias@self.w1)
        Z_bias = np.hstack((np.ones((N, 1)) * 1, Z))
        Y_hat = sigmoid(Z_bias@self.w2)
        return np.argmax(Y_hat, axis=1)

def main():
    # Carregando os dados
    dados = np.loadtxt("coluna_vertebral.csv", delimiter=",", dtype=str, encoding="utf-8")

    X = dados[:, :-1].astype(float)  
    Y_dados = dados[:, -1]  
    rotulos = {"NO": 0, "DH": 1, "SL": 2}
    Y = np.array([rotulos[y] for y in Y_dados])
    Y_codificado = one_hot_encode(Y)

    # Normalização Min-Max
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_normalizado = (X - X_min) / (X_max - X_min)

    rodadas = 100
    particionamento = .8
    max_epoca = 500
    taxa_aprendizagem = 0.05
    precisao = 0.001

    acuracias, sensibilidades, especificidades = [], [], []
    matrizes_confusao = []
    n_amostras = X.shape[0]
    n_treino = int(particionamento * n_amostras)

    # Simulações por Monte Carlo
    for i in range(rodadas):
        # Embaralhar
        indices = np.random.permutation(n_amostras)
        indices_treino, indices_teste = indices[:n_treino], indices[n_treino:]

        X_treino, Y_treino = X_normalizado[indices_treino], Y_codificado[indices_treino]
        X_teste, Y_teste = X_normalizado[indices_teste], Y_codificado[indices_teste]


        # Fase de treinamento e teste
        mlp = MLP(X.shape[1], 3, 3, taxa_aprendizagem)
        mlp.treino(X_treino, Y_treino, max_epoca, precisao)
        Y_pred = mlp.predizer(X_teste)

        Y_real = np.argmax(Y_teste, axis=1)
        
        # Calcular métricas 
        matriz = matriz_confusao(Y_real, Y_pred, num_classes=3)
        acuracia, sensibilidade, especificidade = calcular_metricas(matriz)
        
        acuracias.append(acuracia)
        sensibilidades.append(sensibilidade)
        especificidades.append(especificidade)
        matrizes_confusao.append(matriz)

    # Estatísticas
    print(f"Média Acurácia: {np.mean(acuracias):.4f} | Desvio: {np.std(acuracias):.4f}")
    print(f"Média Sensibilidade: {np.mean(sensibilidades):.4f}")
    print(f"Média Especificidade: {np.mean(especificidades):.4f}")

    # Gráficos
    media = np.mean(acuracias)
    desvio = np.std(acuracias)
    maximo = np.max(acuracias)
    minimo = np.min(acuracias)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

    # Matrizes de Confusão - Melhor e Pior
    idx_max, idx_min = np.argmax(acuracias), np.argmin(acuracias)

    for idx, titulo, cor in [(idx_max, "Maior Acurácia", "Blues"), (idx_min, "Menor Acurácia", "Reds")]:
        matriz = matrizes_confusao[idx]

        plt.figure(figsize=(5, 4))
        sns.heatmap(matriz, annot=True, cmap=cor, fmt="d")
        plt.title(f"{titulo} ({acuracias[idx]:.4f})")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()
    
if __name__ == "__main__":
    main()