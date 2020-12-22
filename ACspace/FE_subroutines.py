""" This file includes the necessary modules for
implementing AC mixed-FEM. 
See TODD ARBOGAST, MAICON R. CORREA, SIAM journal 2016 
"""

import numpy as np
import math
import sys
from sympy import *

def PiolaTransform(coord_E, Xhat):
    """ This function maps Xhat=(xhat,yhat) in Ehat=[-1,1]^2 
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
               [x3,y3]]

    Xhat = [[xhat],[yhat]] vector in Ehat
    Output:
    ------
    X=[[x],[y]]: mapped vector in E.
    DF_E: Jacobian matrix
    J_E: det(DF_E)
    """
    xhat = Xhat[0][0]
    yhat = Xhat[1][0]
    P = 0.25 * np.array([[(1-xhat)*(1-yhat),
                          (1+xhat)*(1-yhat),
                          (1+xhat)*(1+yhat),
                          (1-xhat)*(1+yhat)]])
    # x=F_E(xhat)
    X =np.transpose(np.matmul(P, coord_E))
    # gradient of P, 1st row = dP/dxhat, 2nd row=dP/dyhat
    GradP = 0.25 * np.array([[-(1-yhat),(1-yhat),(1+yhat),-(1+yhat)],
                             [-(1-xhat),-(1+xhat),(1+xhat),(1-xhat)]])

    # DF_E = [[dx/dxhat, dx/dyhat],
    #         [dy/dxhat, dy/dyhat]]
    DF_E = np.transpose(np.matmul(GradP, coord_E))
    J_E = np.linalg.det(DF_E)

    # now we can transform vector vhat to v by eq. (3.4)
    # v(x) = P_E(vhat)(x) = (DF_E/J_E)*vhat(xhat)

    return X, DF_E, J_E

def Vspace(x, y):

    a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
    vx  = a1*x + b1*y + g1 + r*x**2 + 2*s*x*y
    vy  = a2*x + b2*y + g2 - 2*r*x*y - s*y**2
    V = [vx, vy]
    return V

def ACbasis(coord_E):
    
    basis = []
    for i in range(4):
        for j in range(2):
            k = 2*i+j
            eqs = [ np.dot(Vspace(coord_E[0][0], coord_E[0][1]),[-1, 0]),
                    np.dot(Vspace(coord_E[0][0], coord_E[0][1]),[ 0,-1]),
                    np.dot(Vspace(coord_E[1][0], coord_E[1][1]),[ 1, 0]),
                    np.dot(Vspace(coord_E[1][0], coord_E[1][1]),[ 0,-1]),
                    np.dot(Vspace(coord_E[2][0], coord_E[2][1]),[-1, 0]),
                    np.dot(Vspace(coord_E[2][0], coord_E[2][1]),[ 0, 1]),
                    np.dot(Vspace(coord_E[3][0], coord_E[3][1]),[ 1, 0]),
                    np.dot(Vspace(coord_E[3][0], coord_E[3][1]),[ 0, 1])]
            eqs[k] -= 1
            sol = solve(eqs)
            x, y = symbols('x y')    
            V = Vspace(x,y) 
            vx = V[0]
            vy = V[1]
            ux = vx.subs(sol)
            uy = vy.subs(sol)

            basis.append(ux)
            basis.append(uy)



    return basis
coord_E = [[-1.,-1.],
           [1.,-1.],
           [-1.,1.],
           [1.,1.]]
basis = ACbasis(coord_E)
print(basis[15])

