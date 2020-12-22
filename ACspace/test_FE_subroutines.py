import unittest
import FE_subroutines as FE
import numpy as np

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


def main():
    unittest.main()


if __name__ == '__main__':
    main()    