import numpy as np
def NormalInverseCDF(x): #computes inverse Normal CDF using Moro algorithm. Defined for x in (0,1)
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0) or np.any(x >= 1):
        raise ValueError("Input values must be in the range (0, 1).")
    a = [
    2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637,
    ]

    b = [
    -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833
    ]

    c = [
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187
    ]

    y = x - 0.5
    zeros = np.zeros_like(y)
    out = np.zeros_like(y)
    mask_central = np.abs(y) < 0.42 # mask for central region |y| < 0.42
    num = zeros[mask_central]
    den = zeros[mask_central]
    r_central = np.square(y[mask_central])
    for i in range(len(a)):
        num += a[i] * (r_central**i)
        den += b[i] * (r_central**(i+1))
    num *= y[mask_central]
    den += 1
    out[mask_central] = num/den

    mask_tail = np.abs(y) >= 0.42 # mask for tail region |y| >= 0.42
    r_tail = np.where(y[mask_tail]<0,x[mask_tail],1-x[mask_tail])
    s = np.log(-np.log(r_tail))
    t = np.zeros_like(s)
    for i in range(len(c)):
        t += c[i] * (s**i)
    out[mask_tail] = np.where(x[mask_tail]>0.5,t,-t)
    return out

def NormalCDF(x): #Compute Normal CDF
    x = np.asarray(x, dtype=float)
    a = [1.330274429, 
        -1.821255978, 
        1.781477937, 
        -0.356563782, 
        0.31938153]
    mask_left = x<0 #save indices where x is negative
    x = np.abs(x) #now all values are positive. We will process the result for negative x appropriately later
    p = np.zeros_like(x)
    k = 1/(1 + 0.2316419*x)
    for num in a:
        p += num
        p *= k
    p *= (-1/np.sqrt(2*np.pi))*np.exp(-np.square(x)/2)
    p += 1
    p[mask_left] = 1 - p[mask_left] # using N(x) = 1 - N(-x) for x < 0
    return p



