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
        Xhat = [[-1],
                [-1]]
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
        Xhat = [[0.],
                [0.]]
        X, DF_E, J_E = FE.PiolaTransform(coord_E, Xhat)

        self.assertEqual(X[0][0], 0.5)
        self.assertEqual(X[1][0], 0.5)
        self.assertEqual(J_E,0.25)
        self.assertEqual(DF_E[0][0],0.5)
        self.assertEqual(DF_E[0][1],0.)
        self.assertEqual(DF_E[1][1],0.5)


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
        BDM = FE.BDMbasis(coord_E,-1,-1)
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
        AC = FE.ACbasis(coord_E,-1,-1)
        self.assertEqual(np.dot([AC[0],AC[1]],[-1,0]),1.0)
        self.assertEqual(np.dot([AC[2],AC[3]],[0,-1]),1.0)
        self.assertEqual(np.dot([AC[4],AC[5]],[1,0]),0.0)


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


class TestNormal(unittest.TestCase):

    def test_Normal1(self):
        # test on E = [0,0.5]^2
        coord_E = [[0.0,0.0],
                   [0.5,0.0],
                   [0.0,0.5],
                   [0.5,0.5]]
        # check the left edge and (x,y) mapped from (xhat,yhat)
        nl = [-1.,0.]
        n, X = FE.GetNormal(coord_E,-1. ,0.)
        self.assertAlmostEqual(n[0],nl[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nl[1],None,None,1e-8)
        self.assertAlmostEqual(X[1][0],0.25,None,None,1e-8)
        # check the right edge and (x,y) mapped from (xhat,yhat)
        nr = [1.,0]
        n, X = FE.GetNormal(coord_E,1. ,0.)
        self.assertAlmostEqual(n[0],nr[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nr[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.5,None,None,1e-8)
        # check the bottom edge and (x,y) mapped from (xhat,yhat)
        nb = [0.,-1.]
        n, X = FE.GetNormal(coord_E,0 ,-1)
        self.assertAlmostEqual(n[0],nb[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nb[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.25,None,None,1e-8)
        # check the top edge and (x,y) mapped from (xhat,yhat)
        nt = [0.,1.]
        n, X = FE.GetNormal(coord_E,0 ,1)
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
        # check the left edge and (x,y) mapped from (xhat,yhat)
        nl = np.array([-2/math.sqrt(5), 1/math.sqrt(5)])
        n, X = FE.GetNormal(coord_E,-1. ,0.)
        self.assertAlmostEqual(n[0],nl[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nl[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],1/8,None,None,1e-8)
        # check the right edge and (x,y) mapped from (xhat,yhat)
        nr = [3/math.sqrt(10),1/math.sqrt(10)]
        n, X = FE.GetNormal(coord_E,1. ,0.)
        self.assertAlmostEqual(n[0],nr[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nr[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],7/8,None,None,1e-8)
        # check the bottom edge and (x,y) mapped from (xhat,yhat)
        nb = [0.,-1.]
        n, X = FE.GetNormal(coord_E,0 ,-1)
        self.assertAlmostEqual(n[0],nb[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nb[1],None,None,1e-8)
        self.assertAlmostEqual(X[0][0],0.5,None,None,1e-8)
        # check the top edge and (x,y) mapped from (xhat,yhat)
        nt = [-1/math.sqrt(5),2/math.sqrt(5)]
        n, X = FE.GetNormal(coord_E,0. ,1.)
        self.assertAlmostEqual(n[0],nt[0],None,None,1e-8)
        self.assertAlmostEqual(n[1],nt[1],None,None,1e-8)
        self.assertAlmostEqual(X[1][0],0.625,None,None,1e-8)


def main():
    unittest.main()


if __name__ == '__main__':
    main()    