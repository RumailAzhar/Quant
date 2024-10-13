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
    
class Option: #Classes for derivatives, specifying a payoff and time to maturity and other relevant parameters such as strike
    def __init__(self, K = 0, option_type = 'call'):
        #Possible types are 'call', 'put', 'dcall', 'dput', 'forward'. Note dcall and dput refer to digital call and put
        self.K = K
        payoffs = {
            'call': lambda S: np.maximum(0, S-self.K),
            'put': lambda S: np.maximum(0, self.K - S),
            'dcall': lambda S: np.where(S-self.K>0, 1, 0),
            'dput': lambda S: np.where(self.K-S>0, 1, 0),
            'forward': lambda S: S-self.K,
        }
        self.payoff = payoffs[option_type]
        self.option_type = option_type
class GeometricBrownianMotion: #class that simulates GBM of stocks and prices derivatives
    def __init__(self, r=0, d=0, T=0, vol = 0, N_T = 10, S_0 = 1, seed = None):
        self.r = r
        self.d = d
        self.T = T
        self.vol = vol
        self.N_T = N_T #Total steps in simulated paths
        self.S_0 = S_0
        self.seed = seed
    
    def final_values(self, n=1): #Directly find stock value at time T
        rng = RandomNumberGenerator(dimensionality=n, seed = self.seed)
        W = NormalInverseCDF(rng.generate())
        S_values = self.S_0 * np.exp((self.r - self.d)*self.T - 0.5*(self.vol**2)*self.T + self.vol*np.sqrt(self.T)*W)
        return S_values
    
    def simulate_paths(self, n=1): #Store intermedeate stock values at each time step
        dt = self.T/self.N_T #time step
        S_values = np.zeros([n,self.N_T])
        
        for i in range(n):
            if self.seed is None:
                rng = RandomNumberGenerator(dimensionality=self.N_T)
            else:
                rng = RandomNumberGenerator(dimensionality=self.N_T, seed = i + self.seed)
            W = NormalInverseCDF(rng.generate())
            S_values[i,0] = self.S_0
            for j in range(1,self.N_T):
                S_values[i,j]=S_values[i,j-1]*np.exp((self.r - self.d)*dt - 0.5*(self.vol**2)*dt + self.vol*np.sqrt(dt)*W[j])
        return S_values
    
    def simulate_paths_euler(self, n=1): #Euler stepping for path simulation
        dt = self.T/self.N_T #time step
        S_values = np.zeros([n,self.N_T])
        
        for i in range(n):
            if self.seed is None:
                rng = RandomNumberGenerator(dimensionality=self.N_T)
            else:
                rng = RandomNumberGenerator(dimensionality=self.N_T, seed = i + self.seed)
            W = NormalInverseCDF(rng.generate())
            S_values[i,0] = self.S_0
            for j in range(1,self.N_T):
                S_values[i,j]=S_values[i,j-1]*(1 + (self.r - self.d - 0.5*(self.vol**2))*dt + self.vol * np.sqrt(dt) * W[j])
        return S_values
    
    def plot_paths(self, n=1, method = 'exact'): #Plot full paths of stock price
        dt = self.T/self.N_T
        if method == 'exact':
            S_values = self.simulate_paths(n=n)
        if method == 'euler':
            S_values = self.simulate_paths_euler(n=n)
        t_values = np.arange(0, self.T , dt)
        for i in range(n):
            plt.plot(t_values, S_values[i,:])
        plt.show()

    def monte_carlo_price(self, option = None, final_values = np.array([0]), return_variance = False): 
        discounted_payoff = np.exp(-self.r * self.T) * option.payoff(final_values)
        expected_payoff = np.mean(discounted_payoff) #discounted expectation of payoffs gives the risk neutral price
        if not return_variance:
            return expected_payoff
        else:
            sample_variance = np.mean(np.square(discounted_payoff)) - np.square(expected_payoff)
            standard_error = np.sqrt(sample_variance/len(final_values))
            return expected_payoff, sample_variance, standard_error #returning estimated price, sample variance and standard error

    

