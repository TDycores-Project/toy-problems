import unittest
import FE_subroutines2 as FE
import numpy as np
import math

class TestPiolaTransformation(unittest.TestCase):

    def test_Piola1(self):
        """ Note 
        3---2
        |   |
        0---1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [1.,1.],
                   [0.,1.]]
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
        3---2
        |   |
        0---1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [1.,1.],
                   [0.,1.]]
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
                   [0.5,0.5],
                   [0.0,0.5]]
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
                   [0.75,0.75],
                   [0.25,0.5]]
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
        3-------4    
        |       |     
        1-------2
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [1.,0.25],
                   [0.,0.25]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v11.nb=1 and v42.nl=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v12.nb=1 and v21.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v22.nr=1 and v31.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),1.0,None,None,1e-10)
        # check other nodes 
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)

        # check node 4,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v32.nt=1 and v41.nl=1 
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)


class TestNodalBasisNonUniform(unittest.TestCase):

    #check Nhat, the Nodal basis on uniform mesh
    def test_GetNodalBasis2(self):
        """ element E is taken from fig 3.5 of Zhen Tao PhD thesis
                 4
                  \
           3       \ 
          /         \
         /           \
        1-------------2
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.75,0.75],
                   [0.25,0.5]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v11.nb=1 and v42.nl=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v12.nb=1 and v21.nr=1 
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v22.nr=1 and v31.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),1.0,None,None,1e-10)
        # check other nodes 
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)

        # check node 4,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # Note: Nhat = [v11,v12,v21,v22,v31,v32,v41,v42]
        # check v32.nt=1 and v41.nl=1 
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nl),1.0,None,None,1e-10)
        # check other nodes
        self.assertAlmostEqual(np.dot(Nhat[:,0],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nl),0.0, None, None,1e-10)


class TestDivNodalBasis(unittest.TestCase):

    #check Dhat the divergence of Nodal basis on non-uniform mesh
    def test_GetDivNodalBasis(self):
        """ element E is taken from fig 3.5 of Zhen Tao PhD thesis
                 4
                  \
           3       \ 
          /         \
         /           \
        1-------------2
        """
        coord_E = np.array([[0.,0.],
                            [1.,0.],
                            [0.75,0.75],
                            [0.25,0.5]])
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        Dhat = FE.GetDivACNodalBasis(coord_E)

        normals = np.block([[nb],[nb],[nr],[nr],[nt],[nt],[nl],[nl]])
        nodes = np.block([coord_E[0,:],coord_E[1,:],coord_E[1,:],coord_E[2,:],
                          coord_E[2,:],coord_E[3,:],coord_E[3,:],coord_E[0,:]])

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
        IEN1 = np.array([[0, 1],
                         [3, 4],
                         [5, 6],
                         [2, 3]])

        IEN = FE.GetConnectivity(2,1)
        for i in range(4):
            for j in range(2):
                self.assertEqual(IEN1[i][j], IEN[i][j])

    def test_connectivity2(self):
        IEN1 = np.array([[0, 3],
                         [2, 5],
                         [3, 6],
                         [1, 4]])

        IEN = FE.GetConnectivity(1,2)
        for i in range(4):
            for j in range(2):
                self.assertEqual(IEN1[i][j], IEN[i][j])

    def test_connectivity3(self):
            IEN1 = np.array([[0, 1, 5, 6],
                             [3, 4, 8, 9],
                             [5, 6, 10,11],
                             [2, 3, 7, 8]])

            IEN = FE.GetConnectivity(2,2)
            for i in range(4):
                for j in range(4):
                    self.assertEqual(IEN1[i][j], IEN[i][j])

    def test_connectivity4(self):
            IEN1 = np.array([[0, 1, 2, 7, 8, 9],
                             [4, 5, 6, 11, 12, 13],
                             [7, 8, 9, 14, 15, 16],
                             [3, 4, 5, 10, 11, 12]])

            IEN = FE.GetConnectivity(3,2)
            for i in range(4):
                for j in range(6):
                    self.assertEqual(IEN1[i][j], IEN[i][j])


class TestGetNodeCoord(unittest.TestCase):

    def testGetNodeCoord(self):
        xx = np.array([[0], [1], [0], [1]])
        yy = np.array([[0], [0], [1], [1]])
        x, y = FE.GetNodeCoord(1, 1)
        # the output is x (4x1 array) and y (4x1 array)
        for i in range(4):
            self.assertEqual(x[i][0], xx[i][0])
            self.assertEqual(y[i][0], yy[i][0])

    def testGetNodeCoord2(self):
        xx = np.array([[0], [0.5], [1], [0], [0.5], [1]])
        yy = np.array([[0], [0], [0], [1], [1], [1]])
        x, y = FE.GetNodeCoord(2, 1)
        # the output is x (4x1 array) and y (4x1 array)
        for i in range(6):
            self.assertEqual(x[i][0], xx[i][0])
            self.assertEqual(y[i][0], yy[i][0])


def main():
    unittest.main()


if __name__ == '__main__':
    main() 