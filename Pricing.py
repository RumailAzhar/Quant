from math import exp, log, sqrt
from NormalFunctions import NormalCDF

def ForwardPrice(S,K,T,r,d):
    P=exp((r-d)*T)*S - K
    P*=exp(-r*T)
    return P

def CallPrice(S,K,T,r,d,vol):
    d_1 = (log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    C = S*exp(-d*T)*NormalCDF(d_1) - K*exp(-r*T)*NormalCDF(d_2)
    return C

def PutPrice(S,K,T,r,d,vol):
    d_1 = (log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    P = -S*exp(-d*T)*NormalCDF(-d_1) + K*exp(-r*T)*NormalCDF(-d_2)
    return P

def BondPrice(T,r):
    return exp(-r*T)

def DigitalCallPrice(S,K,T,r,d,vol):
    d_1 = (log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    return exp(-r*T)*NormalCDF(d_2)

def DigitalPutPrice(S,K,T,r,d,vol):
    d_1 = (log(S/K) + T*(r-d + 0.5*(vol**2)))/(vol*sqrt(T))
    d_2 = d_1 - vol*sqrt(T)
    return exp(-r*T)*NormalCDF(-d_2)

