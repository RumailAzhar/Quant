from math import exp, log, sqrt
from NormalFunctions import NormalCDF
import numpy as np
def ForwardPrice(S=0,K=0,T=0,r=0,d=0):
    P=exp((r-d)*T)*S - K
    P*=exp(-r*T)
    return P

def CallPrice(S=0,K=0,T=0,r=0,d=0,vol=0):
    d_1 = (np.log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    C = S*exp(-d*T)*NormalCDF(d_1) - K*exp(-r*T)*NormalCDF(d_2)
    return C

def PutPrice(S=0,K=0,T=0,r=0,d=0,vol=0):
    d_1 = (np.log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    P = -S*exp(-d*T)*NormalCDF(-d_1) + K*exp(-r*T)*NormalCDF(-d_2)
    return P

def BondPrice(T=0,r=0):
    return exp(-r*T)

def DigitalCallPrice(S=0,K=0,T=0,r=0,d=0,vol=0):
    d_1 = (np.log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    return exp(-r*T)*NormalCDF(d_2)

def DigitalPutPrice(S=0,K=0,T=0,r=0,d=0,vol=0):
    d_1 = (np.log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    return exp(-r*T)*NormalCDF(-d_2)

