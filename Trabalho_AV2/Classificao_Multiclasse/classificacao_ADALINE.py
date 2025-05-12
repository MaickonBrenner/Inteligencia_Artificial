import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Adaline:
    def __init__(self, X_treino, Y_treino, max_epoca, taxa_aprendizagem, precisao, plot=True):
        self.p,self.N = X_treino.shape
        self.X_treino = np.vstack((
            -np.ones((1,self.N)),
            X_treino
        ))
        self.taxa_aprendizagem = taxa_aprendizagem
        self.Y_treino = Y_treino
        self.precisao = precisao
        self.max_epoca = max_epoca
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))
        self.plot = plot
        self.x1 = np.linspace(-2,7)
        if self.plot:
            plt.figure(1)
            plt.scatter(X_treino[0,:5],X_treino[1,:5],c='purple')
            plt.scatter(X_treino[0,5:],X_treino[1,5:],c='b')
            self.linha = plt.plot(self.x1,self.__gerar_reta(),c='k')
            plt.xlim(-.5,6.5)
            plt.ylim(-.5,6.5)

    def treino(self):
        epocas = 0
        EQM1 = 1
        EQM2 = 0
        hist_eqm = []
        while epocas < self.max_epoca and abs(EQM1 - EQM2) > self.precisao:
            EQM1 = self.__EQM(self.X_treino)
            hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_treino[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                d_k = float(self.Y_treino[k])
                e_k = d_k - u_k
                self.w = self.w + self.taxa_aprendizagem*e_k*x_k # Função de aprendizagem
            if(self.plot):
                plt.pause(.1)
                self.linha[0].remove()
                self.linha = plt.plot(self.x1,self.__gerar_reta(),c='k')

            EQM2 = self.__EQM(self.X_treino)
            epocas+=1
        print(epocas)
        if self.plot:
            plt.figure(2)
            plt.plot(hist_eqm)
            plt.xlabel("Épocas")
            plt.ylabel("EQM")
            plt.show()

    def predizer(self, X_teste):
        X_teste = np.vstack((
            -np.ones((1, X_teste.shape[1])),
            X_teste))
        previsoes = self.w.T@X_teste
        return  np.where(previsoes >= 0, 1, -1)

    def __EQM(self,X):
        eqm = 0
        p_1,N = X.shape
        for k in range(N):
            x_k = X[:,k].reshape(p_1,1)
            u_k = (self.w.T@x_k)
            d_k = float(self.Y_treino[k])
            eqm += (d_k - u_k)**2
        
        return eqm/(2*N)
    
    def __gerar_reta(self):
        return np.nan_to_num(-self.x1*self.w[1,0]/self.w[2,0] + self.w[0,0]/self.w[2,0]) # Equação de Reta

def main():
    # Carregando os dados
    dados = np.loadtxt("coluna_vertebral.csv", delimiter=",", dtype=str, encoding="utf-8")

    X = dados[:, :-1].astype(float)  
    Y = dados[:, -1]  

    codificacao = {
        "NO": np.array([1, -1, -1]), # Normal
        "DH": np.array([-1, 1, -1]), # Hérnia de Disco
        "SL": np.array([-1, -1, 1])  # Espondilolistese
    }

    Y_codificado = np.array([codificacao[rotulo] for rotulo in Y])

    # Normalização Min-Max Scaling
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normalizado = (X - X_min) / (X_max - X_min)

    rodadas = 10
    particionamento = .8
    max_epoca = 100
    taxa_aprendizagem = 0.01
    precisao = 0.001

    acuracias = []

    n_amostras = X_normalizado.shape[0]
    indices = np.arange(n_amostras)
    np.random.shuffle(indices)
    
    n_treino = int(n_amostras * particionamento)
    indices_treino = indices[:n_treino]
    indices_teste = indices[n_treino:]

    X_treino, Y_treino = X_normalizado[indices_treino], Y_codificado[indices_treino]
    X_teste, Y_teste = X_normalizado[indices_teste], Y_codificado[indices_teste]

    # Simulações por Monte Carlo
    for i in range(rodadas):
        # Embaralhar
        np.random.shuffle(indices)
        indices_treino = indices[:n_treino]
        indices_teste = indices[n_treino:]

        X_treino, Y_treino = X_normalizado[indices_treino], Y_codificado[indices_treino]
        X_teste, Y_teste = X_normalizado[indices_teste], Y_codificado[indices_teste]

        # Fase de treinamento        
        adaline = Adaline(X_treino, Y_treino, max_epoca, taxa_aprendizagem, precisao)
        adaline.treino()

        # Teste do modelo
        Y_pred = adaline.predizer(X_teste)

        # Calcular métricas
        VP, VN, FP, FN = matriz_confusao(Y_teste[:, 0], Y_pred)
        acuracia, sensibilidade, especificidade = calcular_metricas(VP, VN, FP, FN)

        acuracias.append(acuracia)

if __name__ == "__main__":
    main()