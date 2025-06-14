import numpy as np
import matplotlib.pyplot as plt

class LocalRandomSearch:
    def __init__(self,max_it,sigma,func,restricoes):
        self.max_it = max_it        
        self.f = func
        self.sigma = sigma
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.x_opt = np.random.uniform(low=self.restr_l,high=self.restr_u)
        self.f_opt = self.f(*self.x_opt)
        
        x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        X,Y = np.meshgrid(x_axis,x_axis)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4)
        resposta = self.f(*self.x_opt) #splat operator
        self.ax.scatter(*self.x_opt,resposta,c='r')
        plt.pause(.1)
        
    def perturb(self):
        n = np.random.normal(loc=0,scale=self.sigma,size=self.x_opt.shape)
        x_cand = self.x_opt + n
        for i in range(len(x_cand)):
            if x_cand[i] < self.restr_l[i]:
                x_cand[i] = self.restr_l[i]
            if x_cand[i] > self.restr_u[i]:
                x_cand[i] = self.restr_u[i]
        return x_cand
        
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(*x_cand)
            if f_cand > self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                plt.pause(.1)
            it+=1
        self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*')
        plt.show()
        
class GlobalRandomSearch:
    def __init__(self,max_it,func,restricoes):
        self.max_it = max_it        
        self.f = func
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.x_opt = np.random.uniform(low=self.restr_l,high=self.restr_u)
        self.f_opt = self.f(*self.x_opt)
        
        x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        X,Y = np.meshgrid(x_axis,x_axis)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4)
        resposta = self.f(*self.x_opt) #splat operator
        self.ax.scatter(*self.x_opt,resposta,c='r')
        plt.pause(.1)
    
    def perturb(self):
        return np.random.uniform(low=self.restr_l,high=self.restr_u)
        
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(*x_cand)
            
            self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
            if f_cand > self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                plt.pause(.1)
            it+=1
        self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*')
        plt.show()
        
class HillClimbing:
    def __init__(self,max_it,max_viz,epsilon,func,restricoes):
        self.max_it = max_it
        self.max_viz = max_viz
        self.e = epsilon
        self.f = func
        self.restr_l = restricoes[:,0]
        self.restr_u = restricoes[:,1]
        self.x_opt = restricoes[:,0]
        self.f_opt = self.f(*self.x_opt)
        
        x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
        X,Y = np.meshgrid(x_axis,x_axis)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.plot_surface(X,Y,self.f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4,edgecolors='k')
        resposta = self.f(*self.x_opt) #splat operator
        self.ax.scatter(*self.x_opt,resposta,c='r')
        plt.pause(.1)
       
    def perturb(self):
        x_cand = np.random.uniform(self.x_opt-self.e,self.x_opt+self.e)
        for i in range(len(x_cand)):
            if x_cand[i] < self.restr_l[i]:
                x_cand[i] = self.restr_l[i]
            if x_cand[i] > self.restr_u[i]:
                x_cand[i] = self.restr_u[i]
        return x_cand
            
    def search(self):
        it = 0
        melhoria = True
        while it < self.max_it and melhoria:
            melhoria = False
            for j in range(self.max_viz):
                x_cand = self.perturb()
                f_cand = self.f(*x_cand)
                self.ax.scatter(*x_cand,f_cand,c='purple',alpha=.3)
                if f_cand > self.f_opt:
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    self.ax.scatter(*self.x_opt,self.f_opt,c='r')
                    plt.pause(.1)
                    melhoria = True
                    self.ax
                    break
            it+=1
        self.ax.scatter(*self.x_opt,self.f_opt,c='g',marker='*',s=150)
        plt.show()