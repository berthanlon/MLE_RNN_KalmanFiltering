import numpy as np


def pellipse(mu, C, num, c, lstyle='-', lw=2):
    """
    Plots error ellipse for matrix C.
    
    Arguments:
    mu -- vector of mean values
    C -- covariance matrix
    num -- number of points on the ellipse
    col -- color of the ellipse
    c -- scaling factor for the ellipse
    
    Keyword Arguments:
    lstyle -- line style of the ellipse (default '-')
    lw -- line width of the ellipse (default 2)
    
    Returns:
    x1 -- x-coordinates of the ellipse
    x2 -- y-coordinates of the ellipse
    """
    if lstyle is None:
        lstyle = '-'
    if lw is None:
        lw = 2
    
    s1 = C[0, 0]
    s2 = C[1, 1]
    s12 = C[0, 1]
    
    x2 = np.sqrt(s2 * c) * np.linspace(-1+1e-10, 1-1e-10, num)
    x1n = (x2 * s12 - np.sqrt((x2**2 - s2 * c) * (s12**2 - s1 * s2))) / s2
    x1p = (x2 * s12 + np.sqrt((x2**2 - s2 * c) * (s12**2 - s1 * s2))) / s2
    x2 = mu[1] + np.concatenate((x2, x2[::-1]))
    x1 = mu[0] + np.concatenate((x1n, x1p[::-1]))
        
    return x1, x2