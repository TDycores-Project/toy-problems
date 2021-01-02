""" This file includes the necessary modules for
implementing AC mixed-FEM. 
See TODD ARBOGAST, MAICON R. CORREA, SIAM journal 2016 
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

# the following two function is for verifying the BDM space
# written by Nate in HdivSetup/basis.py
def VBDM(Xhat):
    """ 
    See eq. (2.15) of Wheeler, Yotov, SIAM J 2006
    """
    xhat = Xhat[0]
    yhat = Xhat[1]
    a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
    vx  = a1*xhat + b1*yhat + g1 + r*xhat**2 + 2*s*xhat*yhat
    vy  = a2*xhat + b2*yhat + g2 - 2*r*xhat*yhat - s*yhat**2
    V = [vx, vy]
    return V

def BDMbasis(coord_E,Xhat):
    """
    Input:
    ------
    coord_E: is the coordinate of vertices of ref element.
    for BDM ref element is defined on [-1,1]^2
    xhat,yhat: basis is evaluated at (xhat,yhat) in [-1,1]^2
    If you wanna check the basis expression, run
    x, y = symbols('x y')           
    BDM = BDMbasis(coord_E,x,y)  
    Note that vertices here are
    2---3
    |   |
    0---1
    like PETSc
    Output:
    ------
    BDM basis function on ref element [-1,-1]^2 in terms of (x,y)
    """
    nl, X = GetNormal(coord_E, [-1., 0.])
    nr, X = GetNormal(coord_E, [1., 0.])
    nb, X = GetNormal(coord_E, [0., -1.])
    nt, X = GetNormal(coord_E, [0., 1.])
    basis = []
    for i in range(4):
        for j in range(2):
            k = 2*i+j
            eqs = [ np.dot(VBDM([coord_E[0][0], coord_E[0][1]]),nl),
                    np.dot(VBDM([coord_E[0][0], coord_E[0][1]]),nb),
                    np.dot(VBDM([coord_E[1][0], coord_E[1][1]]),nr),
                    np.dot(VBDM([coord_E[1][0], coord_E[1][1]]),nb),
                    np.dot(VBDM([coord_E[2][0], coord_E[2][1]]),nl),
                    np.dot(VBDM([coord_E[2][0], coord_E[2][1]]),nt),
                    np.dot(VBDM([coord_E[3][0], coord_E[3][1]]),nr),
                    np.dot(VBDM([coord_E[3][0], coord_E[3][1]]),nt)]
            eqs[k] -= 1
            sol = solve(eqs)
            V = VBDM(Xhat) 
            vx = V[0]
            vy = V[1]
            ux = vx.subs(sol)
            uy = vy.subs(sol)
            
            basis.append(ux)
            basis.append(uy)

    return basis


def VACred(coord_E, Xhat):
    """ 
    See eq. (3.14), (3.12), (3.15) of AC paper, SIAM J 2016
    Note that (x,y) is defined on E
    and xhat, yhat is defined on Ehat
    """
    xhat = Xhat[0]
    yhat = Xhat[1]
    # sigma_hat_1 = curl((1-xhat^2)*yhat)
    sghat1 = [[1-xhat**2],[2*xhat*yhat]] 
    # for debugging define the supplement as BDM
    # sghat1 = [[xhat**2],[-2*xhat*yhat]] 

    # sigma_hat_2 = curl((1-yhat^2)*xhat)
    sghat2 = [[-2*xhat*yhat],[yhat**2-1]] 
    # for debugging define the supplement as BDM
    #sghat2 = [[2*xhat*yhat],[-yhat**2]]

    X, DF_E, J_E = PiolaTransform(coord_E, Xhat)
    # sigma_i = P_E*sigma_hat_i
    sg1 = (DF_E/J_E) @ sghat1
    sg2 = (DF_E/J_E) @ sghat2

    # (x,y) in E is
    x = X[0][0]
    y = X[1][0]
    a1,a2,b1,b2,g1,g2,r,s = symbols('a1 a2 b1 b2 g1 g2 r s')
    vx  = a1*x + b1*y + g1 + r*sg1[0][0] + s*sg2[0][0]
    vy  = a2*x + b2*y + g2 + r*sg1[1][0] + s*sg2[1][0]
    V = [vx, vy]
    return V

def ACbasis(coord_E,Xhat):
    """
    Input:
    ------
    coord_E: is the coordinate of vertices of element.
    xhat,yhat: coordinate at Ehat
    Note
    2---3
    |   |
    0---1
    Output:
    ------
    ACreduce basis function on element E in terms of (xhat,yhat)
    """
    nl, X = GetNormal(coord_E, [-1., 0.])
    nr, X = GetNormal(coord_E, [1., 0.])
    nb, X = GetNormal(coord_E, [0., -1.])
    nt, X = GetNormal(coord_E, [0., 1.])
    basis = []
    ux_x = []
    uy_y = []
    for i in range(4):
        for j in range(2):
            k = 2*i+j
            eqs = [ np.dot(VACred(coord_E,[-1,-1]),nl),
                    np.dot(VACred(coord_E,[-1,-1]),nb),
                    np.dot(VACred(coord_E,[ 1,-1]),nr),
                    np.dot(VACred(coord_E,[ 1,-1]),nb),
                    np.dot(VACred(coord_E,[-1, 1]),nl),
                    np.dot(VACred(coord_E,[-1, 1]),nt),
                    np.dot(VACred(coord_E,[ 1, 1]),nr),
                    np.dot(VACred(coord_E,[ 1, 1]),nt)]
            eqs[k] -= 1
            sol = solve(eqs)
            # a1,a2,b1,b2,g1,g2,r,s
            sol_val = list(sol.values())
            ux_x.append(sol_val[0])
            uy_y.append(sol_val[3])
            V = VACred(coord_E, Xhat) 
            vx = V[0]
            vy = V[1]
            ux = vx.subs(sol)
            uy = vy.subs(sol)
            basis.append(ux)
            basis.append(uy)

    div = np.array([[(ux_x[0]+ux_x[1]), (uy_y[0]+uy_y[1]), (ux_x[2]+ux_x[3]), (uy_y[2]+uy_y[3]),
                    (ux_x[4]+ux_x[5]), (uy_y[4]+uy_y[5]), (ux_x[6]+ux_x[7]), (uy_y[6]+uy_y[7])]])
    
    N = np.array([[basis[0]+basis[2], 0, basis[4]+basis[6], 0, basis[8]+basis[10], 0, basis[12]+basis[14], 0],
                  [0, basis[1]+basis[3], 0, basis[5]+basis[7], 0, basis[9]+basis[11], 0, basis[13]+basis[15]]])
    
    return basis, div, N


def GetQuadrature(Q, quadmethod):
    """This function returns quadrature points (q) and its weight (w).
    Input:
    ------
    Q: number of quadrature points you want
    quadmethod: The method for computing q and w
    either 'GAUSS' or 'LGL'

    Output:
    -------
    w, q: quadrature weights and quadrature points
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

def GetConnectivity(nelx, nely):

    numelem = nelx*nely
    nodex = nelx + 1
    nodey = nely + 1
    IEN = np.zeros((4,numelem), dtype=int)

    for j in range(0,nely):
        for i in range(0,nelx):
            ele = (j)*nelx + i
            
            IEN[0][ele] = i + j*nodex
            IEN[1][ele] = i + j*nodex + 1
            IEN[2][ele] = i + j*nodex + nodex
            IEN[3][ele] = i + j*nodex + nodex + 1

    return IEN


def GetNodeCoord(nelx, nely):
    """ This function returns the physical coordinates of the nodes.
    Input:
    ------
    nelx:   integer
            number of elements in the x direction.
    nely:   integer
            number of elements in the y direction.
    Output:
    -------
    x:      float (1d array)
            the coordinate of the node in the x direction
    y:      float (1d array)
            the coordinate of the node in the y direction
    The geometry we are working on is like the following.
    (for nelx = 2, nely = 2)
    6---------7----------8
    |         |   (3)    |
    |   (2)   |      ----5
    |      ---4-----/    |
    3-----/   |   (1)    |
    |         |      ----2
    |   (0)   |     /
    |     ----1----/
    0----/
    There are 4 elements (numbering in parenthesis), and 9 nodes.
    Bottom edge (0 to 1) is y=0.5x^2. (see src/test_subroutines.py)
    This function returns x,y as 9x2 array for the above mesh.
    """
    nodex = nelx + 1
    nodey = nely + 1
    numnodes = nodex*nodey
    
    # Divide [0,1] by nodex (mesh in the x direction)
    x0 = np.linspace(0, 1, nodex)
    y0 = 0.0 * x0**2               # the bottom geometry line

    y = np.zeros((numnodes, 1))
    for i in range(0, nodex):
        # Divide [0,1] by nodey (mesh in the y direction)
        y1 = np.linspace(y0[i], 1, nodey)
        for j in range(0, nodey):
            y[i + j*nodex] = y1[j]   # collection of y

    x = np.zeros((numnodes, 1))
    for i in range(0, nodey):
        for j in range(0, nodex):
            x[j + i*nodex] = x0[j]   # collection of x

    return x, y


def Perm():

    K = np.array([[1,0],[0,1]])
    return K


def GetElementMat(Q, quadmethod, nelx, nely, e):
    """create
    Me = In.T*W*In, where In is (2Q,8) shape matrix, Q is the number of quadrature
    Be
    """
    IEN = GetConnectivity(nelx, nely)
    x , y = GetNodeCoord(nelx, nely)
    # get coordinate of the element
    ce = np.zeros((4,1), dtype=int)
    for i in range(4):
        ce[i][0] = IEN[i][e]
    coord_E = np.array([[x[ce[0,0]][0], y[ce[0,0]][0]],
                        [x[ce[1,0]][0], y[ce[1,0]][0]],
                        [x[ce[2,0]][0], y[ce[2,0]][0]],
                        [x[ce[3,0]][0], y[ce[3,0]][0]]])

    print(coord_E)
    print(x)
    print(y)
    w, q = GetQuadrature(Q,quadmethod)
    K = Perm()
    Kinv = np.linalg.inv(K)
    In = np.zeros((0,8))
    Dn = np.zeros((0,8))
    W = np.zeros((2*Q*Q,2*Q*Q))
    Np = np.array([[1]])
    NNp = np.zeros((0,1))
    Fp = np.zeros((0,1))
    W2 = np.zeros((Q*Q,Q*Q))
    for i in range(0,Q):
        for j in range(0,Q):
            xhat = q[j]
            yhat = q[i]
            ww = w[i]*w[j]
            X, DF_E, J_E = PiolaTransform(coord_E, [xhat,yhat])
            x = X[0][0]
            y = X[1][0]
            fp = np.array([[2*(math.pi)**2*math.sin(math.pi*x)*math.sin(math.pi*y)]])
            AC, div, N = ACbasis(coord_E,[xhat,yhat])
            In = np.append(In,N, axis=0)
            W[2*j+2*Q*i][2*j+2*Q*i]=Kinv[0][0]*ww*J_E
            W[2*j+2*Q*i][2*j+1+2*Q*i]=Kinv[0][1]*ww*J_E
            W[2*j+1+2*Q*i][2*j+2*Q*i]=Kinv[1][0]*ww*J_E
            W[2*j+1+2*Q*i][2*j+1+2*Q*i]=Kinv[1][1]*ww*J_E
            W2[j+Q*i][j+Q*i] = ww
            Dn = np.append(Dn,div,axis=0)
            NNp = np.append(NNp,Np,axis=0)
            Fp = np.append(Fp,J_E*fp,axis=0)

    Me = In.T @ W @ In
    Be = NNp.T @ W2 @ Dn
    Fep = NNp.T @ W2 @ Fp
    Feu = np.zeros((8,1))

    Ke = np.block([[Me, Be.T],[Be, 0*Np]])
    Fe = np.block([[Feu],[Fep]])
    return Fe, Ke


def GetID_LM(nelx, nely):

    nodex = nelx + 1
    nodey = nely + 1
    numnodes = nodex*nodey
    numelem = nelx*nely
    ndof_u = 8
    ID = np.zeros((2,numnodes), dtype=int)
    for i in range(0,numnodes):
        for j in range(0,2):
            ID[j][i] = 2*i + j
    IEN = GetConnectivity(nelx, nely)
    LMu = np.zeros((ndof_u,numelem), dtype=int)
    for i in range(0,numelem):
        for j in range(0,4):
            idd1 = ID[0][IEN[j][i]]
            idd2 = ID[1][IEN[j][i]]
            LMu[2*j][i] = idd1
            LMu[2*j+1][i] = idd2

    # add pressure dof to LM
    ndof_p = 1
    maxLMu = np.amax(LMu)
    LMp = np.zeros((ndof_p,numelem), dtype=int)
    for i in range(0,numelem):
        LMp[0][i] = maxLMu + i + 1

    LM = np.block([[LMu],[LMp]])

    return ID, LM


def Assembly(Q, quadmethod, nelx, nely):

    ID, LM = GetID_LM(nelx, nely)
    numelem = nelx*nely
    ndof = np.amax(LM) + 1
    neldof = 9
    temp = np.zeros((neldof,1),dtype=int)
    K = np.zeros((ndof,ndof))
    F = np.zeros((ndof,1))
    for e in range(numelem):
        Fe, Ke = GetElementMat(Q, quadmethod, nelx, nely, e)
        temp[:,0] = LM[:,e]
        for i in range(neldof):
            I = temp[i,0]
            if I>-1:
                F[I,0] = F[I,0] + Fe[i,0]
                for j in range(neldof):
                    J = temp[j,0]
                    if J>-1:
                        K[I,J] = K[I,J] + Ke[i,j]

    return F, K
