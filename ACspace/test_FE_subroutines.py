import unittest
import FE_subroutines as FE

class TestPiolaTransformation(unittest.TestCase):

    def test_Piola1(self):
        coord_E = [[0.,0.],
                   [1.,0.],
                   [1.,1.],
                   [0.,1.]]
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
        coord_E = [[0.,0.],
                   [1.,0.],
                   [1.,1.],
                   [0.,1.]]
        Xhat = [[0.],
                [0.]]
        X, DF_E, J_E = FE.PiolaTransform(coord_E, Xhat)

        self.assertEqual(X[0][0], 0.5)
        self.assertEqual(X[1][0], 0.5)
        self.assertEqual(J_E,0.25)
        self.assertEqual(DF_E[0][0],0.5)
        self.assertEqual(DF_E[0][1],0.)
        self.assertEqual(DF_E[1][1],0.5)


def main():
    unittest.main()


if __name__ == '__main__':
    main()    