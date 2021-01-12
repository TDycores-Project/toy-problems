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
    """ Test of this function is passed! See test_FE_subroutines.py
    This function maps Xhat=[xhat,yhat] in Ehat=[-1,1]^2 
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
    """Test of this function is passed! See test_FE_subroutines.py
    This function returns the normal n and coordinate (x,y) on edge in element E.
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
    returns 2x8 matrix which is our 8 prime basis phi_j
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

def VondermondeMat(coord_E):
    """
    Input:
    ------
    coord_E: is the coordinate of vertices of element.
    Note
    3---4
    |   |
    1---2
    Output:
    ------
    VM: the 8x8 vondermonde matrix
    """
    nl, X = GetNormal(coord_E, [-1., 0.])
    nr, X = GetNormal(coord_E, [1., 0.])
    nb, X = GetNormal(coord_E, [0., -1.])
    nt, X = GetNormal(coord_E, [0., 1.])
    normals = np.block([[nl],[nb],[nr],[nb],[nl],[nt],[nr],[nt]])
    nodes = np.block([[-1,-1],[-1,-1],[1,-1],[1,-1],[-1,1],[-1,1],[1,1],[1,1]])
    # vondermonde matrix, V_ij = phi_j(x_i).n_i
    VM = np.zeros((8,8))

    for i in range(8):
        for j in range(8):
            V = VACred(coord_E, nodes[i,:])
            VM[i,j] = np.dot(V[:,j],normals[i,:])

    return VM


def GetACNodalBasis(coord_E,Xhat):
    """This function returns the AC Nodal basis at point Xhat=[xhat,yhat]
    Input:
    ------
    coord_E: coordinate of element E as 4x2 array
    Xhat: is the coordinate at reference element [-1,1]^2

    Output:
    -------
    Nhat: the nodal basis computed at Xhat=[xhat,yhat]
    shape (2,8) as
    Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
    """
    VM = VondermondeMat(coord_E)
    V = VACred(coord_E, Xhat)
    invVM = np.linalg.inv(VM)
    Nhat = V @ invVM

    return Nhat


def GetDivACNodalBasis(coord_E):
    """This function returns the divergence of AC Nodal basis at point Xhat=[xhat,yhat]
    Input:
    ------
    coord_E: coordinate of element E as 4x2 array
    Xhat: is the coordinate at reference element [-1,1]^2

    Output:
    -------
    Dhat: the divergence of nodal basis computed at Xhat=[xhat,yhat]
    shape (1,8)
    """
    VM = VondermondeMat(coord_E)
    # This is the divergence of prime basis given
    # in VACred functions
    divV = np.array([[0, 1, 0, 0, 0, 1, 0, 0]])
    invVM = np.linalg.inv(VM)
    Dhat = divV @ invVM

    return Dhat


def GetQuadrature(Q, quadmethod):
    """Test of this function is passed! See test_FE_subroutines.py
    This function returns quadrature points (q) and its weight (w).
    Input:
    ------
    Q: number of quadrature points you want in 1D
    quadmethod: The method for computing q and w
    either 'GAUSS' or 'LGL'

    Output:
    -------
    w, q: quadrature weights and quadrature points in 1D
    """

    if quadmethod=='GAUSS':
        beta = []
        alpha = range(1, Q)
        for i in range(0,Q-1):
            beta1 = 0.5/math.sqrt(1-(2*alpha[i])**(-2))
            beta.append(beta1)

        D, V = np.linalg.eig(np.diag(beta, k=1) + np.diag(beta, k=-1))
        # we need to sort the eigenvalues, is not sorted
        idx = np.argsort(D)
        V = V[:,idx]
        w = []
        q = []
        for i in range(0,Q):
            w.append(2*V[0][i]**2)
            q.append(D[idx[i]])

    elif quadmethod=='LGL':
        x = []
        alpha = range(0, Q)
        for i in range(0,Q):
            x.append(-math.cos(math.pi*alpha[i]/(Q-1)))

        x = np.array(x)
        P = np.zeros((Q,Q))
        xold = 2*np.ones(Q)
        error = np.absolute(x-xold)
        iteration = 0
        for i in range(0,Q):
            while (error[i] > 1e-16):
                xold = x
                
                P[:,0] = 1
                P[:,1] = x
                for k in range(1,Q-1):
                    P[:,k+1] = np.divide(((2*k+1)*(x*P[:,k]) - (k)*P[:,k-1]),k+1)

                x = xold - np.divide(x*P[:,Q-1]-P[:,Q-2],Q*P[:,Q-1])
                error = np.absolute(x-xold)
                iteration+=1

        w=np.divide(2,(Q-1)*Q*P[:,Q-1]**2)
        q=x
    else:
        print("Error: quadmethod is wrong!Please enter 'GAUSS' or 'LGL'!")
        sys.exit(1)
    
    return w, q


def Perm():
    """ permeability is consider 1 as given
    in MMS of AC paper.
    """
    k = np.array([[1,0],[0,1]])
    return k

def GetMassMat(coord_E,Q,quadmethod):
    """This function returns the interpolation matrix at quadrature points
    N and mass matrix Me = N^T*W*N, where
    W = diag(W1,W2,...Wq) and Wi = wi*Kinv where Kinv is inverse of permeability matrix
    Wi is 2x2 matrix.

    Input:
    ------
    coord_E: coordinate of element E as 4x2 array
    Q: number of quadrature points you want in 1D
    quadmethod: The method for computing q and w
    either 'GAUSS' or 'LGL'

    Output:
    -------
    N: Nodal basis evaluated at quadrature points
    shape of N is (2*Q*Q,8) again Q is quadrature points in 1D
    Me: the nodal interpolation matrix computed at quadrature points
    shape (8,8), 
    """
    k = Perm()
    kinv = np.linalg.inv(k)
    w, q = GetQuadrature(Q, quadmethod)
    N = np.zeros((0,8))
    W = np.zeros((2*Q*Q,2*Q*Q))
    for i in range(Q):
            for j in range(Q):
                xhat = q[j]
                yhat = q[i]
                ww = w[i]*w[j]
                Nhat = GetACNodalBasis(coord_E, [xhat,yhat])
                N = np.append(N,Nhat, axis=0)
                W[2*j+2*Q*i][2*j+2*Q*i]=kinv[0][0]*ww
                W[2*j+2*Q*i][2*j+1+2*Q*i]=kinv[0][1]*ww
                W[2*j+1+2*Q*i][2*j+2*Q*i]=kinv[1][0]*ww
                W[2*j+1+2*Q*i][2*j+1+2*Q*i]=kinv[1][1]*ww

    Me = N.T @ W @ N

    return N, Me

def GetDivMat(coord_E,Q,quadmethod):
    """This function returns the interpolation matrix at quadrature points
    N and mass matrix Me = N^T*W*N, where
    W = diag(W1,W2,...Wq) and Wi = wi*Kinv where Kinv is inverse of permeability matrix
    Wi is 2x2 matrix.

    Input:
    ------
    coord_E: coordinate of element E as 4x2 array
    Q: number of quadrature points you want in 1D
    quadmethod: The method for computing q and w
    either 'GAUSS' or 'LGL'

    Output:
    -------
    N: Nodal basis evaluated at quadrature points
    shape of N is (2*Q*Q,8) again Q is quadrature points in 1D
    Me: the nodal interpolation matrix computed at quadrature points
    shape (8,8), 
    """
    w, q = GetQuadrature(Q, quadmethod)
    D = np.zeros((0,8))
    Nhatp = np.array([[1]])
    Np = np.zeros((0,1))
    W = np.zeros((Q*Q,Q*Q))
    for i in range(Q):
            for j in range(Q):
                xhat = q[j]
                yhat = q[i]
                ww = w[i]*w[j]
                Dhat = GetDivACNodalBasis(coord_E)
                D = np.append(D,Dhat, axis=0)
                W[j+Q*i][j+Q*i] = ww
                Np = np.append(Nhatp,Np,axis=0)

    Be = Np.T @ W @ D

    return Be
