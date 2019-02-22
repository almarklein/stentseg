""" Functions for statistics

Author Maaike Koenrades
"""

import scipy
from scipy import stats
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats


def normality_check(x, a=None, reta=False, alpha=0.05, showhist=True):
    """ Check for normality by Shapiro Wilk and histogram plot
        x is an array of sample data (1D of shape (n,) or (n,1))
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


def paired_samples_ttest(a,b, profile='Name', alpha=0.05, amplitude=False):
    """ Use scipy stats for a paired sampled Students t-test
    profile: name of analysis
    returns calculated t-statistic and two-tailed p-value
    """
    if amplitude:
        a = np.asarray(a)
        b = np.asarray(b)
    else:
        if isinstance(a,list):
            a = a[0]
        if isinstance(b,list):
            b = b[0]
        
    t2, p2 = stats.ttest_rel(a,b)
    print(profile +": t = " + str(t2))
    print(profile +": p = " + str(p2))
    if p2 < alpha:
        print('yes different')
    else:
        print('no difference')
    print("mean difference: " + str(np.mean(a-b)))
    print()
    
    return t2, p2

def independent_samples_ttest(a, b, ):
    """ Use scipy stats for a paired sampled Students t-test
    returns calculated t-statistic and two-tailed p-value
    """
    
    t2, p2 = stats.ttest_ind(a,b)
    print(profile +": t = " + str(t2))
    print(profile +": p = " + str(p2))
    if p2 < alpha:
        print('yes different')
    else:
        print('no difference')
    print("mean difference: " + str(np.mean(a-b)))
    print()
    
    return t2, p2
    
