import unittest
import FE_subroutines as FE
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


class Testbasis(unittest.TestCase):

    def test_BDMbasis1(self):
        """ Note 
        2---3
        |   |
        0---1
        """
        coord_E = [[-1.,-1.],
                   [1.,-1.],
                   [-1.,1.],
                   [1.,1.]]
        BDM = FE.BDMbasis(coord_E,[-1,-1])
        self.assertEqual(np.dot([BDM[0],BDM[1]],[-1,0]),1.0)
        self.assertEqual(np.dot([BDM[2],BDM[3]],[0,-1]),1.0)
        self.assertEqual(np.dot([BDM[4],BDM[5]],[1,0]),0.0)

    def test_ACbasis1(self):
        """ Note 
        2---3
        |   |
        0---1
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.,1.],
                   [1.,1.]]
        AC, div, N = FE.ACbasis(coord_E,[-1,-1])
        self.assertEqual(np.dot([AC[0],AC[1]],[-1,0]),1.0)
        self.assertEqual(np.dot([AC[2],AC[3]],[0,-1]),1.0)
        self.assertEqual(np.dot([AC[4],AC[5]],[1,0]),0.0)

    def test_ACbasis2(self):
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
        nl = np.array([-2/math.sqrt(5), 1/math.sqrt(5)])
        nr = [3/math.sqrt(10),1/math.sqrt(10)]
        nt = [-1/math.sqrt(5),2/math.sqrt(5)]
        nb = [0.,-1.]
        # check node 0,
        AC, div, N = FE.ACbasis(coord_E,[-1,-1])
        # check v11.nl=1 and v12.nb=1
        self.assertAlmostEqual(np.dot([AC[0],AC[1]],nl),1.0,None,None,1e-10)
        #print("v11=",[AC[0],AC[1]])
        self.assertAlmostEqual(np.dot([AC[2],AC[3]],nb),1.0,None,None,1e-10)
        #print("v12=",[AC[2],AC[3]])
        # check v21.nr = 0
        self.assertAlmostEqual(np.dot([AC[4],AC[5]],nr),0.0, None, None,1e-10)

        # check node 1,
        AC, div, N = FE.ACbasis(coord_E,[1,-1])
        # check v21.nr=1 and v22.nb=1 
        self.assertAlmostEqual(np.dot([AC[4],AC[5]],nr),1.0,None,None,1e-10)
        #print("v21=",[AC[4],AC[5]])
        self.assertAlmostEqual(np.dot([AC[6],AC[7]],nb),1.0,None,None,1e-10)
        #print("v22=",[AC[6],AC[7]])
        # check v11.nl = 0
        self.assertAlmostEqual(np.dot([AC[0],AC[1]],nl),0.0, None, None,1e-10)

        # check node 2,
        AC, div, N = FE.ACbasis(coord_E,[-1,1])
        # check v31.nl=1 and v32.nt=1 
        self.assertAlmostEqual(np.dot([AC[8],AC[9]],nl),1.0,None,None,1e-10)
        #print("v31=",[AC[8],AC[9]])
        self.assertAlmostEqual(np.dot([AC[10],AC[11]],nt),1.0,None,None,1e-10)
        #print("v32=",[AC[10],AC[11]])
        # check v41.nr = 0
        self.assertAlmostEqual(np.dot([AC[12],AC[13]],nr),0.0, None, None,1e-10)

        # check node 3,
        AC, div, N = FE.ACbasis(coord_E,[1,1])
        # check v41.nr=1 and v42.nt=1 
        self.assertAlmostEqual(np.dot([AC[12],AC[13]],nr),1.0,None,None,1e-10)
        #print("v41=",[AC[12],AC[13]])
        self.assertAlmostEqual(np.dot([AC[14],AC[15]],nt),1.0,None,None,1e-10)
        #print("v42=",[AC[14],AC[15]])
        # check v31.nl = 0
        self.assertAlmostEqual(np.dot([AC[8],AC[9]],nl),0.0, None, None,1e-10)

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

    def test_connectivity(self):
        IEN1 = np.array([[0, 1, 3, 4],
                         [1, 2, 4, 5],
                         [3, 4, 6, 7],
                         [4, 5, 7, 8]])

        IEN2 = np.array([[ 0,  1,  2,  4,  5,  6],
                         [ 1,  2,  3,  5,  6,  7],
                         [ 4,  5,  6,  8,  9, 10],
                         [ 5,  6,  7,  9, 10, 11]])
        IEN = FE.GetConnectivity(2,2)
        for i in range(2):
            for j in range(2):
                self.assertEqual(IEN1[i][j], IEN[i][j])

        IEN = FE.GetConnectivity(3,2)
        for i in range(3):
            for j in range(2):
                self.assertEqual(IEN2[i][j], IEN[i][j])


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