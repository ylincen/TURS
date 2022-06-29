# cython: profile=True

# This script is used to calculate the regret for multinomial model and histogram model
# All code below are re-written based on the RCpp code from the 'SCCI' R package

from numpy import log, log2, sqrt, ceil
import cython

def regret(M: cython.long, K: cython.long) -> cython.double:
    if K > 100:
        alpha =  K / M
        ca = 0.5 + 0.5 * sqrt(1.0 + 4.0/alpha)
        logReg = M * (log(alpha) + (alpha + 2.0) * log(ca) - 1.0 / ca) - 0.5 * log(ca + 2.0 / alpha)
        return logReg / log(2.0)
    else:
        costs = regretPrecal(M, K)
        if costs <= 0.0:
            return 0.0
        else:
            return myLog2(costs)

def myLog2(v: cython.double) -> cython.double:
    if v == 0.0:
        return 0.0
    else:
        return log2(v)

def binaryRegretPrecal(M: cython.long) -> cython.double:
    i: cython.long
    b: cython.double
    p = 10
    if M < 1:
        return 0.0
    sum = 1.0
    b = 1.0
    bound = int(ceil(2.0 + sqrt(2.0 * M * p * log(10.0))))
    for i in range(1, bound + 1):
        b = (M - i + 1) * (b / M)
        sum += b
    return sum


def regretPrecal(M: cython.long, K: cython.long) -> cython.double:
    if K < 1:
        return 0.0
    elif K == 1:
        return 1.0
    else:
        sum = binaryRegretPrecal(M)
        old_sum = 1.0
        if K > 2:
            for j in range(3, K + 1):
                new_sum = sum + (M * old_sum) / (j - 2.0)
                old_sum = sum
                sum = new_sum
        return sum