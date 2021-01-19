import unittest
import FEACSubroutines as FE
import numpy as np
import math

class TestPiolaTransformation(unittest.TestCase):

    def test_Piola1(self):
        """ Note 
        2---3
        |   |
        0---1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.,1.],
                   [1.,1.]]
        Xhat = [-1,-1]
        X, DF_E, J_E = FE.PiolaTransform(coord_E, Xhat)

        self.assertEqual(X[0][0], coord_E[0][0])
        self.assertEqual(X[1][0], coord_E[0][1])
        self.assertEqual(J_E,0.25)
        self.assertEqual(DF_E[0][0],0.5)
        self.assertEqual(DF_E[0][1],0.)
        self.assertEqual(DF_E[1][1],0.5)

    def test_Piola2(self):
        """ Note 
        2---3
        |   |
        0---1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.,1.],
                   [1.,1.]]
        Xhat = [0., 0.]
        X, DF_E, J_E = FE.PiolaTransform(coord_E, Xhat)

        self.assertEqual(X[0][0], 0.5)
        self.assertEqual(X[1][0], 0.5)
        self.assertEqual(J_E,0.25)
        self.assertEqual(DF_E[0][0],0.5)
        self.assertEqual(DF_E[0][1],0.)
        self.assertEqual(DF_E[1][1],0.5)


class TestNormal(unittest.TestCase):

    def test_Normal1(self):
        # test on E = [0,0.5]^2
        coord_E = [[0.0,0.0],
                   [0.5,0.0],
                   [0.0,0.5],
                   [0.5,0.5]]
        # check the left edge and (x,y) mapped from (xhat,yhat)
        nl = [-1.,0.]
        n, X = FE.GetNormal(coord_E,[-1. ,0.])
        self.assertAlmostEqual(n[0],nl[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nl[1],None,None,1e-8)
        self.assertAlmostEqual(X[1][0],0.25,None,None,1e-8)
        # check the right edge and (x,y) mapped from (xhat,yhat)
        nr = [1.,0]
        n, X = FE.GetNormal(coord_E,[1. ,0.])
        self.assertAlmostEqual(n[0],nr[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nr[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.5,None,None,1e-8)
        # check the bottom edge and (x,y) mapped from (xhat,yhat)
        nb = [0.,-1.]
        n, X = FE.GetNormal(coord_E,[0 ,-1])
        self.assertAlmostEqual(n[0],nb[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nb[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.25,None,None,1e-8)
        # check the top edge and (x,y) mapped from (xhat,yhat)
        nt = [0.,1.]
        n, X = FE.GetNormal(coord_E,[0 ,1])
        self.assertAlmostEqual(n[0],nt[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nt[1],None,None,1e-8)
        self.assertAlmostEqual(X[1][0],0.5,None,None,1e-8)


class TestNormal2(unittest.TestCase):
    # test based on Fig 3.5 of Zhen Tao PhD thesis
    def test_Normal2(self):
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.25,0.5],
                   [0.75,0.75]]
        # check the left edge normal and middle point (x,y) mapped from (xhat,yhat)
        nl = np.array([-2/math.sqrt(5), 1/math.sqrt(5)])
        # enter the middle point (-1,,0) on the left edge of Ehat
        n, X = FE.GetNormal(coord_E,[-1. ,0.])
        self.assertAlmostEqual(n[0],nl[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nl[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],1/8,None,None,1e-8)

        # check the right edge normal and middle point (x,y) mapped from (xhat,yhat)
        nr = [3/math.sqrt(10),1/math.sqrt(10)]
        # enter the middle point (1,0) on the right edge of Ehat
        n, X = FE.GetNormal(coord_E,[1. ,0.])
        self.assertAlmostEqual(n[0],nr[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nr[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],7/8,None,None,1e-8)

        # check the bottom edge normal and middle point (x,y) mapped from (xhat,yhat)
        nb = [0.,-1.]
        # enter the middle point (0,-1) on the bottom edge of Ehat
        n, X = FE.GetNormal(coord_E,[0 ,-1])
        self.assertAlmostEqual(n[0],nb[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nb[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.5,None,None,1e-8)

        # check the top edge normal and middle point (x,y) mapped from (xhat,yhat)
        nt = [-1/math.sqrt(5),2/math.sqrt(5)]
        # enter the middle point (0,1) on the top edge of Ehat
        n, X = FE.GetNormal(coord_E,[0. ,1.])
        self.assertAlmostEqual(n[0],nt[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nt[1],None,None,1e-8)
        self.assertAlmostEqual(X[1][0],0.625,None,None,1e-8)


class TestNodalBasisUniform(unittest.TestCase):

    #check Nhat, the Nodal basis on uniform mesh
    def test_GetNodalBasis1(self):
        """ element E 
        2-------3    
        |       |     
        0-------1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.,0.25],
                   [1.,0.25]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 0,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v0.nb=1 and v4.nl=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v1.nb=1 and v2.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v5.nl=1 and v6.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),1.0,None,None,1e-10)
        # check other nodes 
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v7.nt=1 and v3.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)


class TestNodalBasisNonUniform(unittest.TestCase):

    #check Nhat, the Nodal basis on uniform mesh
    def test_GetNodalBasis2(self):
        """ element E is taken from fig 3.5 of Zhen Tao PhD thesis
                 3
                  \
           2       \ 
          /         \
         /           \
        0-------------1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.25,0.5],
                   [0.75,0.75]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 0,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v0.nb=1 and v4.nl=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v1.nb=1 and v2.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v5.nl=1 and v6.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),1.0,None,None,1e-10)
        # check other nodes 
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # Note: Nhat = [v0,v1,v2,v3,v4,v5,v6,v7]
        # check v7.nt=1 and v3.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nt),0.0, None, None,1e-10)


class TestDivNodalBasisUniform(unittest.TestCase):

    #check Dhat the divergence of Nodal basis on uniform mesh
    def test_GetDivNodalBasis1(self):
 
        coord_E = np.array([[0.0,0.5],
                            [1.0,0.5],
                            [0.0,1.0],
                            [1.0,1.0]])
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        Dhat = FE.GetDivACNodalBasis(coord_E)

        normals = np.block([[nb],[nb],[nr],[nr],[nl],[nl],[nt],[nt]])
        nodes = np.block([coord_E[0,:],coord_E[1,:],coord_E[1,:],coord_E[3,:],
                          coord_E[0,:],coord_E[2,:],coord_E[2,:],coord_E[3,:]])

        # test with u = [x-y,x+y] ==>div(u) = 2
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([x-y,x+y],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],2.0, None, None,1e-10)

        # test with u = [-y,x] ==>div(u) = 0
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([-y,x],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],0.0, None, None,1e-10)

        # test with u = [-x,x] ==>div(u) = -1
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([-x,x],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],-1.0, None, None,1e-10)


class TestDivNodalBasisNonUniform(unittest.TestCase):

    #check Dhat the divergence of Nodal basis on non-uniform mesh
    def test_GetDivNodalBasis(self):
        """ element E is taken from fig 3.5 of Zhen Tao PhD thesis
                 3
                  \
           2       \ 
          /         \
         /           \
        0-------------1
        """
        coord_E = np.array([[0.,0.],
                            [1.,0.],
                            [0.25,0.5],
                            [0.75,0.75]])
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        Dhat = FE.GetDivACNodalBasis(coord_E)

        normals = np.block([[nb],[nb],[nr],[nr],[nl],[nl],[nt],[nt]])
        nodes = np.block([coord_E[0,:],coord_E[1,:],coord_E[1,:],coord_E[3,:],
                          coord_E[0,:],coord_E[2,:],coord_E[2,:],coord_E[3,:]])

        # test with u = [x-y,x+y] ==>div(u) = 2
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([x-y,x+y],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],2.0, None, None,1e-10)

        # test with u = [-y,x] ==>div(u) = 0
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([-y,x],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],0.0, None, None,1e-10)

        # test with u = [-x,x] ==>div(u) = -1
        u = np.zeros((8,1))
        for i in range(8):
            x = nodes[2*i]
            y = nodes[2*i+1]
            u[i][0] = np.dot([-x,x],normals[i,:])

        const = Dhat @ u
        self.assertAlmostEqual(const[0][0],-1.0, None, None,1e-10)


class TestQuadrature(unittest.TestCase):

    def test_GAUSS1(self):
        delta = 1e-4
        ww2 = [1., 1.]
        qq2 = [-0.5774, 0.5774]
        w, q = FE.GetQuadrature(2,'GAUSS')
        for i in range(0,2):
            self.assertAlmostEqual(ww2[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq2[i], q[i],None,None,delta)
    
    def test_GAUSS2(self):
        delta = 1e-4
        ww3 = [0.5556,  0.8889,  0.5556]
        qq3 = [-0.7746,      0,  0.7746]
        w, q = FE.GetQuadrature(3,'GAUSS')
        for i in range(0,3):
            self.assertAlmostEqual(ww3[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq3[i], q[i],None,None,delta)
    def test_GAUSS3(self):
        delta = 1e-4
        ww5 = [0.2369,    0.4786,    0.5689,    0.4786,    0.2369]
        qq5 = [-0.9062,   -0.5385,    0.0000,    0.5385,    0.9062]
        w, q = FE.GetQuadrature(5,'GAUSS')
        for i in range(0,5):
            self.assertAlmostEqual(ww5[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq5[i], q[i],None,None,delta)
    def test_LGL1(self):
        delta = 1e-4
        ww2 = [1., 1.]
        qq2 = [-1., 1.]
        w, q = FE.GetQuadrature(2,'LGL')
        for i in range(0,2):
            self.assertAlmostEqual(ww2[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq2[i], q[i],None,None,delta)
    
    def test_LGL2(self):
        delta = 1e-4
        ww3 = [0.3333,    1.3333,    0.3333]
        qq3 = [-1.,      0,  1.]
        w, q = FE.GetQuadrature(3,'LGL')
        for i in range(0,3):
            self.assertAlmostEqual(ww3[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq3[i], q[i],None,None,delta)
    def test_LGL3(self):
        delta = 1e-4
        ww5 = [0.1000,    0.5444,    0.7111,    0.5444,    0.1000]
        qq5 = [-1.0000,   -0.6547,         0,    0.6547,    1.0000]
        w, q = FE.GetQuadrature(5,'LGL')
        for i in range(0,5):
            self.assertAlmostEqual(ww5[i], w[i],None,None,delta)
            self.assertAlmostEqual(qq5[i], q[i],None,None,delta)


class TestConnectivity(unittest.TestCase):

    def test_connectivity1(self):
        IEN1e = np.array([[0, 1],
                          [3, 4],
                          [2, 3],
                          [5, 6]])
        IEN1n = np.array([[0, 1],
                          [1, 2],
                          [3, 4],
                          [4, 5]])

        IENe, IENn = FE.GetConnectivity(2,1)
        for i in range(4):
            for j in range(2):
                self.assertEqual(IEN1e[i][j], IENe[i][j])
                self.assertEqual(IEN1n[i][j], IENn[i][j])

    def test_connectivity2(self):
        IEN1e = np.array([[0, 3],
                          [2, 5],
                          [1, 4],
                          [3, 6]])
        IEN1n = np.array([[0, 2],
                          [1, 3],
                          [2, 4],
                          [3, 5]])

        IENe, IENn = FE.GetConnectivity(1,2)
        for i in range(4):
            for j in range(2):
                self.assertEqual(IEN1e[i][j], IENe[i][j])
                self.assertEqual(IEN1n[i][j], IENn[i][j])

    def test_connectivity3(self):
            IEN1e = np.array([[0 , 1 , 5 , 6 ],
                              [3 , 4 , 8 , 9 ],
                              [2 , 3 , 7 , 8 ],
                              [5 , 6 , 10, 11]])

            IEN1n = np.array([[0 , 1 , 3 , 4 ],
                              [1 , 2 , 4 , 5 ],
                              [3 , 4 , 6 , 7 ],
                              [4 , 5 , 7 , 8]])

            IENe, IENn = FE.GetConnectivity(2,2)
            for i in range(4):
                for j in range(4):
                    self.assertEqual(IEN1e[i][j], IENe[i][j])
                    self.assertEqual(IEN1n[i][j], IENn[i][j])

    def test_connectivity4(self):
            IEN1e = np.array([[0, 1, 2, 7 , 8 , 9 ],
                              [4, 5, 6, 11, 12, 13],
                              [3, 4, 5, 10, 11, 12],
                              [7, 8, 9, 14, 15, 16]])
             
            IEN1n = np.array([[0, 1, 2, 4 , 5 , 6 ],
                              [1, 2, 3, 5 , 6 , 7 ],
                              [4, 5, 6, 8 , 9 , 10],
                              [5, 6, 7, 9 , 10, 11]])

            IENe, IENn = FE.GetConnectivity(3,2)
            for i in range(4):
                for j in range(6):
                    self.assertEqual(IEN1e[i][j], IENe[i][j])
                    self.assertEqual(IEN1n[i][j], IENn[i][j])


class TestGetSharedEdgeDof(unittest.TestCase):

    def test_GetSharedEdgeDof1(self):
        nelx = 5
        nely = 1
        edgedof1 = [12, 13, 14, 15, 16, 17, 18, 19]
        edgedof = FE.GetSharedEdgeDof(nelx, nely)
        for i in range(len(edgedof)):
            self.assertEqual(edgedof[i], edgedof1[i])

    def test_GetSharedEdgeDof2(self):
        nelx = 2
        nely = 4
        edgedof1 = [6, 7, 10, 11, 12, 13, 16, 17, 20, 21, 22, 23, 26, 27, 30, 31, 32, 33, 36, 37]
        edgedof = FE.GetSharedEdgeDof(nelx, nely)
        for i in range(len(edgedof)):
            self.assertEqual(edgedof[i], edgedof1[i])

    def test_GetSharedEdgeDof3(self):
        nelx = 2
        nely = 2
        edgedof1 = [6, 7, 10, 11, 12, 13, 16, 17]
        edgedof = FE.GetSharedEdgeDof(nelx, nely)
        for i in range(len(edgedof)):
            self.assertEqual(edgedof[i], edgedof1[i])

    def test_GetSharedEdgeDof4(self):
        nelx = 3
        nely = 2
        edgedof1 = [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25]
        edgedof = FE.GetSharedEdgeDof(nelx, nely)
        for i in range(len(edgedof)):
            self.assertEqual(edgedof[i], edgedof1[i])


class TestDivergenceUniform(unittest.TestCase):
    """
    Test divergence operator on multiple uniform elements
    """
    def test_Divergence1(self):
        nelx = 4
        nely = 1
        U, D = FE.AssembleDivOperator('uniform', nelx, nely)
        Div = D @ U
        # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
        for i in range(len(Div)):
            self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence2(self):
        nelx = 3
        nely = 2
        U, D = FE.AssembleDivOperator('uniform', nelx, nely)
        Div = D @ U
        # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
        for i in range(len(Div)):
            self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence3(self):
            nelx = 2
            nely = 2
            U, D = FE.AssembleDivOperator('uniform', nelx, nely)
            Div = D @ U
            # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
            for i in range(len(Div)):
                self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence4(self):
            nelx = 2
            nely = 5
            U, D = FE.AssembleDivOperator('uniform', nelx, nely)
            Div = D @ U
            # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
            for i in range(len(Div)):
                self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

class TestDivergenceNonUniform(unittest.TestCase):
    """
    Test divergence operator on multiple uniform elements
    """
    def test_Divergence1(self):
        nelx = 3
        nely = 1
        U, D = FE.AssembleDivOperator('nonuniform', nelx, nely)
        Div = D @ U
        # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
        for i in range(len(Div)):
            self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence2(self):
        nelx = 3
        nely = 2
        U, D = FE.AssembleDivOperator('nonuniform', nelx, nely)
        Div = D @ U
        # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
        for i in range(len(Div)):
            self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence3(self):
            nelx = 3
            nely = 3
            U, D = FE.AssembleDivOperator('nonuniform', nelx, nely)
            Div = D @ U
            # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
            for i in range(len(Div)):
                self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)

    def test_Divergence4(self):
            nelx = 2
            nely = 5
            U, D = FE.AssembleDivOperator('nonuniform', nelx, nely)
            Div = D @ U
            # in GetVecUe function velocity is u = [x-y,x+y] ==> div(u) = 2
            for i in range(len(Div)):
                self.assertAlmostEqual(Div[i][0], 2., None, None, 1e-10)


def main():
    unittest.main()


if __name__ == '__main__':
    main() 