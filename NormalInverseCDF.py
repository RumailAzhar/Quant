#Moro Algorithm for Inverse Normal CDF. Section2.1
from math import log
def NormalInverseCDF(x):
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
    if abs(y) < 0.42:
        r = y**2
        nsum = 0
        dsum = 1
        for i, num in enumerate(a):
            nsum += num*(r**i)
        for i, num in enumerate(b):
            dsum += num * (r**(i+1))
        return y*nsum/dsum
    elif y<0:
        r=x
    else:
        r=1-x
    s=log(-log(r))
    t = 0
    for i, num in enumerate(c):
        t += num*(s**i)
    if x>0.5:
        return t
    else:
        return -t
