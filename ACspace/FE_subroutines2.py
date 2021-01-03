""" This file includes the necessary modules for
implementing AC mixed-FEM. 
See TODD ARBOGAST, MAICON R. CORREA, SIAM journal 2016
Here I create the Vondermonde matrix explicitly
"""

import numpy as np
import math
import sys
from sympy import *
import matplotlib.pyplot as plt

def PiolaTransform(coord_E, Xhat):
    """ This function maps Xhat=[xhat,yhat] in Ehat=[-1,1]^2 
    to X=(x,y) in E.
    or vector vhat in Ehat to v in E.
    Written based on eq. (3.3), (3.4) of AC paper 2016
    Input:
    ------
    coord_E: coordinates of quadrilateral E with vertices oriented
    counterclockwise. coord_E is 4x2 array
    coord_E = [[x0,y0],
               [x1,y1],
               [x2,y2],
               [x3,y3]] with vertices numbering
    2----3
    |    |
    0----1
    Xhat = [xhat, yhat] in Ehat
    Output:
    ------
    X=[[x],[y]]: mapped vector in E.
    DF_E: Jacobian matrix
    J_E: det(DF_E)
    later we transform vector vhat to v by eq. (3.4)
    v(x) = P_E(vhat)(x) = (DF_E/J_E)*vhat(xhat)
    """
    xhat = Xhat[0]
    yhat = Xhat[1]
    P = 0.25 * np.array([[(1-xhat)*(1-yhat),
                          (1+xhat)*(1-yhat),
                          (1-xhat)*(1+yhat),
                          (1+xhat)*(1+yhat)]])
    P_E = P @ coord_E
    # x=F_E(xhat)
    X = P_E.T
    # gradient of P, 1st row = dP/dxhat, 2nd row=dP/dyhat
    GradP = 0.25 * np.array([[-(1-yhat),(1-yhat),-(1+yhat),(1+yhat)],
                             [-(1-xhat),-(1+xhat),(1-xhat),(1+xhat)]])

    # DF_E = [[dx/dxhat, dx/dyhat],
    #         [dy/dxhat, dy/dyhat]]
    JT = GradP @ coord_E
    DF_E = JT.T
    J_E = np.linalg.det(DF_E)

    return X, DF_E, J_E

def GetNormal(coord_E, Xhat):
    """This function returns the normal n and coordinate (x,y) on edge in element E.
    Input:
    ------
    coord_E: vertices coordinate of element E
    (xhat,yhat): coordinate of the edge in element Ehat=[-1,1]^2

    Output:
    -------
    n: computed normal of an edge of element E
    (x,y): mapped point of (xhat, yhat)

    for example if you want normal on the left edge of E
    enter coord_E, and (-1,0) to get 'n' of the left edge of E and corresponding (x,y)
    """

    X, DF_E, J_E = PiolaTransform(coord_E, Xhat)
    
    dxdxhat = DF_E[0][0]
    dydxhat = DF_E[1][0]
    length1 = math.sqrt(dxdxhat*dxdxhat + dydxhat*dydxhat)

    dxdyhat = DF_E[0][1]
    dydyhat = DF_E[1][1]
    length2 = math.sqrt(dxdyhat*dxdyhat + dydyhat*dydyhat)

    xhat = Xhat[0]
    yhat = Xhat[1]
    if (xhat == -1. and -1.<yhat<1.):
        # left edge, (0,0,1)x(dxdyhat,dydyhat,0)
        n = np.array([-dydyhat, dxdyhat])
        n = n/length2

    elif (xhat == 1. and -1.<yhat<1.):
        # right edge, (0,0,-1)x(dxdyhat,dydyhat,0)
        n = np.array([dydyhat, -dxdyhat])
        n = n/length2

    elif (yhat == -1. and -1.<xhat<1.):
        # bottom edge, (0,0,-1)x(dxdxhat,dydxhat,0)
        n = np.array([dydxhat, -dxdxhat])
        n = n/length1

    elif (yhat == 1. and -1.<xhat<1.):
        # top edge, (0,0,1)x(dxdxhat,dydxhat,0)
        n = np.array([-dydxhat, dxdxhat])
        n = n/length1
        #n = [x/length1 for x in n]

    else:
        print("Error! Enter the Xhat=[xhat, yhat] on the edge of Ehat")

    return n, X

def VACred(coord_E, Xhat):
    """ 
    See eq. (3.14), (3.12), (3.15) of AC paper, SIAM J 2016
    Note that (x,y) is defined on E
    and xhat, yhat is defined on Ehat
    """
    xhat = Xhat[0]
    yhat = Xhat[1]
    # sigma_hat_1 = curl((1-xhat^2)*yhat)
    sghat1 = np.array([[1-xhat**2],[2*xhat*yhat]])
 
    # sigma_hat_2 = curl((1-yhat^2)*xhat)
    sghat2 = np.array([[-2*xhat*yhat],[yhat**2-1]])

    X, DF_E, J_E = PiolaTransform(coord_E, Xhat)
    # sigma_i = P_E*sigma_hat_i
    sg1 = (DF_E/J_E) @ sghat1
    sg2 = (DF_E/J_E) @ sghat2

    # (x,y) in E is
    x = X[0][0]
    y = X[1][0]
    # we have 8 basis including sg1 and sg2
    v1 = np.array([[1],[0]])
    v2 = np.array([[x],[0]])
    v3 = np.array([[y],[0]])
    v4 = np.array([[0],[1]])
    v5 = np.array([[0],[x]])
    v6 = np.array([[0],[y]])

    V = np.block([v1,v2,v3,v4,v5,v6,sg1,sg2])

    return V
