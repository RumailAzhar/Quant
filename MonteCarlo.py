import numpy as np
import matplotlib.pyplot as plt
from NormalFunctions import NormalInverseCDF
class RandomNumberGenerator:
    def __init__(self, seed=None, dimensionality=1):
        self.dimensionality = dimensionality # n = number of random samples
        self.seed = seed

    def generate(self):
        rng = np.random.default_rng(seed = self.seed) # PCG64 generator
        return rng.random(size = self.dimensionality) #returns numpy array of random uniform numbers in [0,1)

class GeometricBrownianMotion:
    def __init__(self, r=0, d=0, T=0, vol = 0, dt = 0.001, S_0 = 1):
        self.r = r
        self.d = d
        self.T = T
        self.vol = vol
        self.dt = dt
        self.S_0 = S_0
    
    def final_values(self, n=1):
        S_values = np.zeros(n)
        rng = RandomNumberGenerator(dimensionality=n)
        W = (rng.generate())
        for i, num in enumerate(W):
            W[i] = NormalInverseCDF(num)
        for i in range(n):
            S = self.S_0
            S*=np.exp((self.r - self.d)*self.T - 0.5*(self.vol**2)*self.T + self.vol*np.sqrt(self.T)*W[i])
            S_values[i] = S
        return S_values
    
    def simulate_paths(self, n=1):
        N_t = int(self.T/self.dt + 1) #length of random walk array
        S_values = np.zeros([n,N_t])
        
        for i in range(n):
            rng = RandomNumberGenerator(dimensionality=N_t)
            W = rng.generate()
            for k in range(N_t):
                W[k] = NormalInverseCDF(W[k])
            S_values[i,0] = self.S_0
            for j in range(1,N_t):
                S_values[i,j]=S_values[i,j-1]*np.exp((self.r - self.d)*self.dt - 0.5*(self.vol**2)*self.dt + self.vol*np.sqrt(self.dt)*W[j])
        return S_values
    
    def plot_paths(self, n=1):
        S_values = self.simulate_paths(n=n)
        t_values = np.arange(0, self.T + self.dt, self.dt)
        for i in range(n):
            plt.plot(t_values, S_values[i,:])
        plt.show()


