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
    4---3
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
    normals = np.block([[nb],[nb],[nr],[nr],[nl],[nl],[nt],[nt]])
    nodes = np.block([[-1,-1],[1,-1],[1,-1],[1,1],[-1,-1],[-1,1],[-1,1],[1,1]])
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
    Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
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
                Np = np.append(Np,Nhatp,axis=0)

    Be = Np.T @ W @ D

    return Be

def GetConnectivity(nelx, nely):
    """This function returns the connectivity array based on edge
    ----5--------6----
    |       |        |
    2       3        4        
    |       |        |
    ----0--------1----

    3-------4--------5
    |       |        |
    |       |        |
    |       |        |
    0-------1--------2
    local numbering of one element is
    ----3----
    |       |
    2       1
    |       |
    ----0----

    2-------3
    |       |
    |       |
    |       |
    0-------1


    Input:
    ------
    nelx: number of element in x direction start from 1 NOT 0
    nely: number of element in y direction start from 1 NOT 0
    Output:
    ------
    IENe: connectivity array of size 4x(nelx*nely) based on edge numbering
    We need IENe for assembly
    IENn: connectivity array of size 4x(nelx*nely) based on node numbering
    We need IENn for find the coordinate of nodes in assembly
    """
    # number of element
    numelem = nelx*nely
    # number of nodes in x-direction
    nodex = nelx + 1
    # number of nodes in y-direction
    nodey = nely + 1
    IENe = np.zeros((4,numelem), dtype=int)
    for j in range(0,nely):
        for i in range(0,nelx):
            ele = (j)*nelx + i
            
            IENe[0][ele] = i + j*(nodex+nelx)
            IENe[1][ele] = i + j*(nodex+nelx) + nodex
            IENe[2][ele] = i + j*(nodex+nelx) + nelx
            IENe[3][ele] = i + j*(nodex+nelx) + (nodex+nelx)

    IENn = np.zeros((4,numelem), dtype=int)
    for j in range(0,nely):
        for i in range(0,nelx):
            ele = (j)*nelx + i
            IENn[0][ele] = i + j*nodex
            IENn[1][ele] = i + j*nodex + 1
            IENn[2][ele] = i + j*nodex + nodex
            IENn[3][ele] = i + j*nodex + nodex + 1

    return IENe, IENn

def GetNodeCoord(mesh, nelx, nely):
    """ This function returns the physical coordinates of the nodes.
    Input:
    ------
    nelx:   integer
            number of elements in the x direction.
    nely:   integer
            number of elements in the y direction.
    mesh: can be unifrom or nonuniform

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
    if mesh == 'uniform':
        y0 = 0.0 * x0              # the bottom geometry line
    else:
        y0 = 0.5 * x0

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


def GetID(nelx, nely):
    """
    This function returns an array of global dof of size (2,numedges)
    each edge has 2 dof
    """

    nodex = nelx + 1
    nodey = nely + 1
    numedges = (nodex*nely) + (nodey*nelx)
    ID = np.zeros((2,numedges), dtype=int)
    for i in range(0,numedges):
        for j in range(0,2):
            ID[j][i] = 2*i + j

    return ID


def GetLMu(nelx, nely):
    """
    This function is LM for velocity created based on ID arrany and connectivity
    """

    numelem = nelx*nely
    ndof_u = 8

    ID = GetID(nelx, nely)
    IENe, IENn = GetConnectivity(nelx, nely)

    LMu = np.zeros((ndof_u,numelem), dtype=int)
    for i in range(0,numelem):
        for j in range(0,4):
            idd1 = ID[0][IENe[j][i]]
            idd2 = ID[1][IENe[j][i]]
            LMu[2*j][i] = idd1
            LMu[2*j+1][i] = idd2

    return LMu

def GetLMp(nelx, nely):
    """
    This is LMp for pressure, maybe not needed
    """
    # add pressure dof to LM
    LMu = GetLMu(nelx, nely)
    maxLMu = np.amax(LMu)
    ndof_p = 1
    numelem = nelx*nely
    LMp = np.zeros((ndof_p,numelem), dtype=int)
    for i in range(0,numelem):
        LMp[0][i] = maxLMu + i + 1

    return LMp

def GetCoordElem(mesh, nelx, nely, e):
    """
    This functions returns coordinate of element "e" 
    """
    IENe, IENn = GetConnectivity(nelx, nely)
    x , y = GetNodeCoord(mesh, nelx, nely)
    # get coordinate of the element
    ce = np.zeros((4,1), dtype=int)
    CoordElem = np.zeros((4,2))
    for i in range(4):
        ce[i][0] = IENn[i][e]
        CoordElem[i][0] = x[ce[i,0]][0]
        CoordElem[i][1] = y[ce[i,0]][0]

    return CoordElem


def GetGlobalNormal(mesh, nelx, nely, e):
    """
    This function returns the global dof normals.
    Note that the local dof directions are outward.
    Then we compare the global dof direction and
    local dof direction for assembly later

    ----3----
    |       |
    2       1
    |       |
    ----0----

    2-------3
    |       |
    |       |
    |       |
    0-------1
    """
    IENe, IENn = GetConnectivity(nelx, nely)
    x , y = GetNodeCoord(mesh, nelx, nely)
    # get coordinate of the element
    ce = np.zeros((4,1), dtype=int)
    for i in range(4):
        ce[i][0] = IENn[i][e]
    
    x0 = x[ce[0,0]][0]
    x1 = x[ce[1,0]][0]
    x2 = x[ce[2,0]][0]
    x3 = x[ce[3,0]][0]
    y0 = y[ce[0,0]][0]
    y1 = y[ce[1,0]][0]
    y2 = y[ce[2,0]][0]
    y3 = y[ce[3,0]][0]

    # Get Tangential direction form node i to j, i< j as + direction
    Tb = np.array([x1 - x0, y1 - y0])
    Lb = math.sqrt(Tb[0]**2 + Tb[1]**2)
    Tb = Tb/Lb
    Tr = np.array([x3 - x1, y3 - y1])
    Lr = math.sqrt(Tr[0]**2 + Tr[1]**2)
    Tr = Tr/Lr
    Tl = np.array([x2 - x0, y2 - y0])
    Ll = math.sqrt(Tl[0]**2 + Tl[1]**2)
    Tl = Tl/Ll
    Tt = np.array([x3 - x2, y3 - y2])
    Lt = math.sqrt(Tt[0]**2 + Tt[1]**2)
    Tt = Tt/Lt

    k = np.array([0, 0, 1])
    nb = np.cross(Tb, k)
    Nb = np.array([nb[0], nb[1]])
    nr = np.cross(Tr, k)
    Nr = np.array([nr[0], nr[1]])
    nl = np.cross(Tl, k)
    Nl = np.array([nl[0], nl[1]])
    nt = np.cross(Tt, k)
    Nt = np.array([nt[0], nt[1]])

    return Nb, Nr, Nl, Nt

def GetDivElem(mesh, nelx, nely, e):
    """
    This function returns the divergence operator for each element
    We need it to test assembly 
    """
    CoordElem = GetCoordElem(mesh, nelx, nely, e)
    De = GetDivACNodalBasis(CoordElem)

    return De


def GetVecUe(mesh, nelx, nely, e):
    """
    This function discretize the vector u = [x-y, x+y] on element e
    """
    CoordElem = GetCoordElem(mesh, nelx, nely, e)

    nl, X = GetNormal(CoordElem, [-1., 0.])
    nr, X = GetNormal(CoordElem, [1., 0.])
    nb, X = GetNormal(CoordElem, [0., -1.])
    nt, X = GetNormal(CoordElem, [0., 1.])

    normals = np.block([[nb],[nb],[nr],[nr],[nl],[nl],[nt],[nt]])
    nodes = np.block([CoordElem[0,:],CoordElem[1,:],CoordElem[1,:],CoordElem[3,:],
                      CoordElem[0,:],CoordElem[2,:],CoordElem[2,:],CoordElem[3,:]])

    # test with u = [x-y,x+y] ==>div(u) = 2
    ue = np.zeros((8,1))
    for i in range(8):
        x = nodes[2*i]
        y = nodes[2*i+1]
        u = [x-y, x+ y]
        ue[i][0] = np.dot(u,normals[i,:])

    return ue

def GetElementRestriction(mesh, nelx, nely, e):
    """
    This function is map between local to global dof or element restriction operator.
    We use this function to scatter the local vector or matrix
    to global vector or matrix in assembly process
    """
    LMu = GetLMu(nelx, nely)
    ndof = np.amax(LMu) + 1
    neldof = 8
    temp = np.zeros((neldof,1),dtype=int)
    
    # This is the normal of global dof
    Nb, Nr, Nl, Nt = GetGlobalNormal(mesh, nelx, nely, e)
    CoordElem = GetCoordElem(mesh, nelx, nely, e)
    # This is the normal of local dof
    nl, X = GetNormal(CoordElem, [-1., 0.])
    nr, X = GetNormal(CoordElem, [1., 0.])
    nb, X = GetNormal(CoordElem, [0., -1.])
    nt, X = GetNormal(CoordElem, [0., 1.])
    Loc2Globnormal = np.array([np.dot(nb,Nb), np.dot(nb,Nb), np.dot(nr,Nr), np.dot(nr,Nr),
                               np.dot(nl,Nl), np.dot(nl,Nl), np.dot(nt,Nt), np.dot(nt,Nt)])
    temp[:,0] = LMu[:,e]
    # element restriction operator
    L = np.zeros((neldof,ndof))
    for i in range(neldof):
        if Loc2Globnormal[i] >0:
            # local dof and global dof are in same direction
            L[i][temp[i][0]] = 1
        else:
            # local dof and global dof are in opposite direction
            L[i][temp[i][0]] = -1

    return L 

def GetSharedEdgeDof(nelx, nely):
    """
    This function returns the global dof on shared edge.
    In assembly when we add dof in shared edges, we need to divide it by 2, to avoid
    counting a vector dof twice.
    """
    LMu = GetLMu(nelx, nely)
    numelem = nelx*nely
    neldof = 8
    # get all shared edges
    sharededge1=[]
    for e1 in range(0,numelem):
        for e2 in range(1,numelem):
            for j in range(neldof):
                for i in range(neldof):
                    if LMu[j][e1]==LMu[i][e2] and e1 != e2:
                        sharededge1.append(LMu[j][e1])
    # delete the possible repeated dof           
    sharededge2 = [] 
    [sharededge2.append(x) for x in sharededge1 if x not in sharededge2] 
    idx = np.argsort(sharededge2)
    # sort shared global dof
    sharededge = []
    for i in range(0,len(sharededge2)):
        sharededge.append(sharededge2[idx[i]])

    return sharededge

def AssembleDivOperator(mesh, nelx, nely):

    numelem = nelx*nely
    LMu = GetLMu(nelx, nely)
    ndof = np.amax(LMu) + 1
    U = np.zeros((ndof, 1))
    D = np.zeros((numelem, ndof))
    for e in range(numelem):
        # get element restriction operator L for element e
        L = GetElementRestriction(mesh, nelx, nely, e)
        # get discretized vector u for element e
        Ue = GetVecUe(mesh, nelx, nely, e)
        # get divergence for element e
        De = GetDivElem(mesh, nelx, nely, e)
        # assemble U
        U = U + L.T @ Ue
        # assemble Divergence
        D[e,:] = De @ L

    # divide those repeated dof by 2 in shared edges
    edgedof = GetSharedEdgeDof(nelx, nely)
    for i in range(len(edgedof)):
        U[edgedof[i],0] = U[edgedof[i],0]/2

    return U, D


