""" Check for normality by Shapiro Wilk and histogram plot

Author Maaike Koenrades
"""

import scipy
from scipy import stats
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt 


def normality_check(x, a=None, reta=False, alpha=0.05, showhist=True):
    """ x is an array of sample data (1D of shape (n,) or (n,1))
        W is the test statistic
        a is an Array of internal parameters used in the calculation
        normal distributed when pValue > alpha (null hypothesis not rejected)
    """
    if isinstance(x, list):
        x = np.asarray(x)
    elif x.ndim == 2:
        x = x.flatten() # to 1d
    elif x.ndim > 2:
        raise RuntimeWarning("Array ndim > 2 should be 1-d")
    
    xnonan = x[~np.isnan(x)]
    
    W, pValue = stats.shapiro(xnonan, a, reta)
    
    if pValue > alpha:
        normality = True
    else:
        normality = False
    
    if showhist:
    # histogram plot
        plt.figure()
        plt.clf()
        pyplot.hist(xnonan)
        pyplot.show()
    
    return W, pValue, normality


