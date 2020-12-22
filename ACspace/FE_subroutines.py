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
    later we transform vector vhat to v by eq. (3.4)
    v(x) = P_E(vhat)(x) = (DF_E/J_E)*vhat(xhat)
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

    return X, DF_E, J_E

# the following two function is for verifying the BDM space
# written by Nate in HdivSetup/basis.py
def VBDM(x, y):
    """ 
    See eq. (2.15) of Wheeler, Yotov, SIAM J 2006
    """
    a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
    vx  = a1*x + b1*y + g1 + r*x**2 + 2*s*x*y
    vy  = a2*x + b2*y + g2 - 2*r*x*y - s*y**2
    V = [vx, vy]
    return V

def BDMbasis(coord_E,x,y):
    """
    Input:
    ------
    coord_E: is the coordinate of vertices of ref element.
    for BDM ref element is defined on [-1,-1]^2
    x,y: basis is evaluated at (x,y)
    If you wanna check the basis expression, run
    x, y = symbols('x y')           
    BDM = BDMbasis(coord_E,x,y)  
    Note that vertices here are
    3---4
    |   |
    1---2
    like PETSc
    Output:
    ------
    BDM basis function on ref element [-1,-1]^2 in terms of (x,y)
    """
    basis = []
    for i in range(4):
        for j in range(2):
            k = 2*i+j
            eqs = [ np.dot(VBDM(coord_E[0][0], coord_E[0][1]),[-1, 0]),
                    np.dot(VBDM(coord_E[0][0], coord_E[0][1]),[ 0,-1]),
                    np.dot(VBDM(coord_E[1][0], coord_E[1][1]),[ 1, 0]),
                    np.dot(VBDM(coord_E[1][0], coord_E[1][1]),[ 0,-1]),
                    np.dot(VBDM(coord_E[2][0], coord_E[2][1]),[-1, 0]),
                    np.dot(VBDM(coord_E[2][0], coord_E[2][1]),[ 0, 1]),
                    np.dot(VBDM(coord_E[3][0], coord_E[3][1]),[ 1, 0]),
                    np.dot(VBDM(coord_E[3][0], coord_E[3][1]),[ 0, 1])]
            eqs[k] -= 1
            sol = solve(eqs)
            V = VBDM(x,y) 
            vx = V[0]
            vy = V[1]
            ux = vx.subs(sol)
            uy = vy.subs(sol)

            basis.append(ux)
            basis.append(uy)

    return basis


def VACred(coord_E, x, y, xhat, yhat):
    """ 
    See eq. (3.14), (3.12), (3.15) of AC paper, SIAM J 2016
    Note that (x,y) is defined on E
    and xhat, yhat is defined on Ehat
    """
    # sigma_hat_1 = curl((1-xhat^2)*yhat)
    sghat1 = [[1-xhat^2],[2*xhat*yhat]] 
    # sigma_hat_2 = curl((1-yhat^2)*xhat)
    sghat2 = [[-2*xhat*yhat],[yhat^2-1]] 
    Xhat = [[xhat],[yhat]]
    X, DF_E, J_E = PiolaTransform(coord_E, Xhat)
    # sigma_i = P_E*sigma_hat_i
    sg1 = (DF_E/J_E)*sghat1
    sg2 = (DF_E/J_E)*sghat2

    a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
    vx  = a1*x + b1*y + g1 + r*sg1[0][0] + s*sg2[0][0]
    vy  = a2*x + b2*y + g2 + r*sg1[1][0] + s*sg2[1][0]
    V = [vx, vy]
    return V

def ACbasis(coord_E,x,y):
    """
    Input:
    ------
    coord_E: is the coordinate of vertices of element.
    x,y: basis is evaluated at (x,y)
    Note that vertices here are
    4---3
    |   |
    1---2
    Output:
    ------
    AC reduce basis function on element E in terms of (x,y)
    """
    # I need this for computing V after solve
    xhat = [-1,1,1,-1]
    yhat = [-1,-1,1,1]

    basis = []
    for i in range(4):
        for j in range(2):
            k = 2*i+j
            eqs = [ np.dot(VACred(coord_E,coord_E[0][0], coord_E[0][1],-1,-1),[-1, 0]),
                    np.dot(VACred(coord_E,coord_E[0][0], coord_E[0][1],-1,-1),[ 0,-1]),
                    np.dot(VACred(coord_E,coord_E[1][0], coord_E[1][1],1,-1),[ 1, 0]),
                    np.dot(VACred(coord_E,coord_E[1][0], coord_E[1][1],1,-1),[ 0,-1]),
                    np.dot(VACred(coord_E,coord_E[2][0], coord_E[2][1],1,1),[1, 0]),
                    np.dot(VACred(coord_E,coord_E[2][0], coord_E[2][1],1,1),[ 0, 1]),
                    np.dot(VACred(coord_E,coord_E[3][0], coord_E[3][1],-1,1),[ -1, 0]),
                    np.dot(VACred(coord_E,coord_E[3][0], coord_E[3][1],-1,1),[ 0, 1])]
            eqs[k] -= 1
            sol = solve(eqs)
            V = VACred(coord_E, x, y, xhat[i], yhat[i]) 
            vx = V[0]
            vy = V[1]
            ux = vx.subs(sol)
            uy = vy.subs(sol)
            basis.append(ux)
            basis.append(uy)

    return basis

coord_E = [[-1.,-1.],
           [1.,-1.],
           [1.,1.],
           [-1.,1.]]

AC = ACbasis(coord_E,0,0)
print(AC)

x, y = symbols('x y')
BDM = BDMbasis(coord_E,x,y)
print(BDM)