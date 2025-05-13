import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Adaline:
    def __init__(self, X_treino, Y_treino, max_epoca, taxa_aprendizagem, precisao, plot=False):
        self.p, self.N = X_treino.shape
        self.X_treino = np.vstack((
            -np.ones((1, self.N)),
            X_treino
        ))
        self.taxa_aprendizagem = taxa_aprendizagem
        self.Y_treino = Y_treino
        self.precisao = precisao
        self.max_epoca = max_epoca
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1, 1))
        self.plot = plot
        self.x1 = np.linspace(-2,7)

        if self.plot:
            plt.figure(1)
            plt.scatter(X_treino[0, :], X_treino[1, :], c=['purple' if y == 1 else 'blue' for y in Y_treino])
            self.linha = plt.plot(self.x1,self.__gerar_reta(),c='k')

    def treino(self):
        epocas = 0
        EQM_anterior = 1
        hist_eqm = []

        while epocas < self.max_epoca:
            EQM_atual = self.__EQM()
            hist_eqm.append(EQM_atual)

            if abs(EQM_anterior - EQM_atual) < self.precisao:
                break
            EQM_anterior = EQM_atual

            for k in range(self.N):
                x_k = self.X_treino[:, k].reshape(-1, 1)
                u_k = self.w.T@x_k
                d_k = self.Y_treino[k]
                e_k = d_k - u_k
                self.w += self.w + self.taxa_aprendizagem*e_k*x_k # Função de aprendizagem
            
            if self.plot:
                plt.pause(.1)
                self.linha[0].remove()
                self.linha = plt.plot(self.x1,self.__gerar_reta(),c='k')

            epocas += 1

        if self.plot:
            plt.figure(2)
            plt.plot(hist_eqm)
            plt.xlabel("Épocas")
            plt.ylabel("EQM")
            plt.title("Erro Quadrádo Médio")
            plt.show()

    def predizer(self, X_teste):
        X_teste = np.vstack((
            -np.ones((1, X_teste.shape[1])),
            X_teste))
        saida = self.w.T@X_teste
        return np.where(saida >= 0, 1, -1)

    def __EQM(self):
        erros = [(self.Y_treino[k] - float(self.w.T @ self.X_treino[:, k].reshape(-1, 1)))**2 for k in range(self.N)]
        return np.mean(erros) / 2
    
    def __gerar_reta(self):
        return np.nan_to_num(-self.x1*self.w[1,0]/self.w[2,0] + self.w[0,0]/self.w[2,0]) # Equação de Reta

def matriz_confusao(y_real, y_pred):
    VP = np.sum((y_real == 1) & (y_pred == 1))
    VN = np.sum((y_real == -1) & (y_pred == -1))
    FP = np.sum((y_real == -1) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == -1))
    return VP, VN, FP, FN

def calcular_metricas(VP, VN, FP, FN):
    acuracia = (VP + VN) / (VP + VN + FP + FN)
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0
    return acuracia, sensibilidade, especificidade

def classificacao(X_treino, Y_treino, X_teste, parametros):
    classificadores = []
    resultados = []
    
    # Fase de treinamento
    for i in range(3):
        Y_binario = Y_treino[:, i]
        clf = Adaline(X_treino.T, Y_binario, **parametros)
        clf.treino()
        pred = clf.predizer(X_teste.T)
        classificadores.append(clf)
        resultados.append(pred)
    return np.stack(resultados, axis=1)

def main():
    # Carregando os dados
    dados = np.loadtxt("coluna_vertebral.csv", delimiter=",", dtype=str, encoding="utf-8")

    X = dados[:, :-1].astype(float)  
    Y = dados[:, -1]  

    codificacao = {
        "NO": np.array([1, -1, -1]),
        "DH": np.array([-1, 1, -1]),
        "SL": np.array([-1, -1, 1])
    }

    Y_codificado = np.array([codificacao[y] for y in Y])

    # Normalização Min-Max
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_normalizado = (X - X_min) / (X_max - X_min)

    rodadas = 100
    particionamento = .8
    max_epoca = 100
    taxa_aprendizagem = 0.01
    precisao = 0.001

    acuracias, sensibilidades, especificidades = [], [], []
    n_amostras = X.shape[0]
    n_treino = int(particionamento * n_amostras)

    # Simulações por Monte Carlo
    for i in range(rodadas):
        # Embaralhar
        indices = np.random.permutation(n_amostras)
        indices_treino, indices_teste = indices[:n_treino], indices[n_treino:]

        X_treino, Y_treino = X_normalizado[indices_treino], Y_codificado[indices_treino]
        X_teste, Y_teste = X_normalizado[indices_teste], Y_codificado[indices_teste]

        parametros = {
            "max_epoca": max_epoca,
            "taxa_aprendizagem": taxa_aprendizagem,
            "precisao": precisao,
            "plot": False
        }

        # Fase de treinamento e teste
        Y_pred_bin = classificacao(X_treino, Y_treino, X_teste, parametros)
        Y_pred = np.argmax(Y_pred_bin, axis=1)
        Y_real = np.argmax(Y_teste, axis=1)

        Y_bin_real = (Y_real == 0).astype(int) * 2 - 1
        Y_bin_pred = (Y_pred == 0).astype(int) * 2 - 1
        
        # Calcular métricas
        VP, VN, FP, FN = matriz_confusao(Y_bin_real, Y_bin_pred) 
        acuracia, sensibilidade, especificidade = calcular_metricas(VP, VN, FP, FN)
        
        acuracias.append(acuracia)
        sensibilidades.append(sensibilidade)
        especificidades.append(especificidade)

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
    # Recalcula para índice máximo/mínimo
    for idx, titulo, cor in [(idx_max, "Maior Acurácia", "Blues"), (idx_min, "Menor Acurácia", "Reds")]:
        indices = np.random.permutation(n_amostras)
        idx_treino, idx_teste = indices[:n_treino], indices[n_treino:]
        X_treino, X_teste = X_normalizado[idx_treino], X_normalizado[idx_teste]
        Y_treino, Y_teste = Y_codificado[idx_treino], Y_codificado[idx_teste]
        Y_pred_bin = classificacao(X_treino, Y_treino, X_teste, parametros)
        Y_pred = np.argmax(Y_pred_bin, axis=1)
        Y_true = np.argmax(Y_teste, axis=1)

        Y_bin_true = (Y_true == 0).astype(int) * 2 - 1
        Y_bin_pred = (Y_pred == 0).astype(int) * 2 - 1

        VP, VN, FP, FN = matriz_confusao(Y_bin_true, Y_bin_pred)
        matriz = np.array([[VP, FN], [FP, VN]])

        plt.figure(figsize=(5, 4))
        sns.heatmap(matriz, annot=True, cmap=cor, fmt="d")
        plt.title(f"{titulo} ({acuracias[idx]:.4f})")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.show()
    
if __name__ == "__main__":
    main()