import unittest
import FE_subroutines2 as FE
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


class TestbasisUniform(unittest.TestCase):

    #check the Nodal basis on uniform mesh
    def test_GetNodalBasis1(self):
        """ element E 
        3-------4    
        |       |     
        1-------2
        """
        coord_E = [[0.,0.],
                   [1.,0.],
                   [0.,0.25],
                   [1.,0.25]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # check v11.nl=1 and v12.nb=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        # check other nodes v21.nr = 0,...
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # check v21.nr=1 and v22.nb=1 
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),1.0,None,None,1e-10)
        # check other nodes v11.nl = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # check v31.nl=1 and v32.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),1.0,None,None,1e-10)
        # check other nodes v41.nr = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 4,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # check v41.nr=1 and v42.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),1.0,None,None,1e-10)
        # check other nodes v31.nl = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)


class TestbasisNonUniform(unittest.TestCase):

    #check the Nodal basis on non-uniform mesh
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
                   [0.25,0.5],
                   [0.75,0.75]]
        nl, X = FE.GetNormal(coord_E, [-1., 0.])
        nr, X = FE.GetNormal(coord_E, [1., 0.])
        nb, X = FE.GetNormal(coord_E, [0., -1.])
        nt, X = FE.GetNormal(coord_E, [0., 1.])
        # check node 1,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,-1])
        # check v11.nl=1 and v12.nb=1
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),1.0,None,None,1e-10)
        # check other nodes v21.nr = 0,...
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 2,
        Nhat = FE.GetACNodalBasis(coord_E,[1,-1])
        # check v21.nr=1 and v22.nb=1 
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),1.0,None,None,1e-10)
        # check other nodes v11.nl = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 3,
        Nhat = FE.GetACNodalBasis(coord_E,[-1,1])
        # check v31.nl=1 and v32.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),1.0,None,None,1e-10)
        # check other nodes v41.nr = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),0.0, None, None,1e-10)

        # check node 4,
        Nhat = FE.GetACNodalBasis(coord_E,[1,1])
        # check v41.nr=1 and v42.nt=1 
        self.assertAlmostEqual(np.dot(Nhat[:,6],nr),1.0,None,None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,7],nt),1.0,None,None,1e-10)
        # check other nodes v31.nl = 0
        self.assertAlmostEqual(np.dot(Nhat[:,0],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,1],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,2],nr),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,3],nb),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,4],nl),0.0, None, None,1e-10)
        self.assertAlmostEqual(np.dot(Nhat[:,5],nt),0.0, None, None,1e-10)


def main():
    unittest.main()


if __name__ == '__main__':
    main() 