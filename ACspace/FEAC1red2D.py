""" how to run
python3 FEAC1red2D.py --nelx 3 --nely 3 --Q 3 --quadmethod 'LGL'
if you run with 
python3 FEAC1red2D.py
uses the default values as
nelx=2
nely=2
Q=2
quadmethod='GAUSS'
"""
import FEACSubroutines as FE
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Input for FE_subroutines')

    parser.add_argument('--nelx',
                        dest='nelx',
                        type=int,
                        default=2,
                        help='Number of element in x direction')

    parser.add_argument('--nely',
                        dest='nely',
                        type=int,
                        default=2,
                        help='Number of element in y direction')

    parser.add_argument('--Q',
                        dest='Q',
                        type=int,
                        default=2,
                        help='Number of quadrature points')

    parser.add_argument('--quadmethod',
                        dest='quadmethod',
                        type=str,
                        default='GAUSS',
                        help='Quadmethod which can be GAUSS or LGL')

    parser.add_argument('--mesh',
                        dest='mesh',
                        type=str,
                        default='uniform',
                        help='mesh distribution which can be:uniform, nonuniform, stretched, random')

    args = parser.parse_args()

    nelx = args.nelx
    nely = args.nely
    Q = args.Q
    quadmethod = args.quadmethod
    mesh = args.mesh

    F, K = FE.Assembly(mesh, nelx, nely, Q, quadmethod)

    dp, du = FE.GetFESol(K,F,nelx,nely)
    p, u = FE.GetExactSol(mesh, nelx, nely)

    error_u = np.linalg.norm(du-u)
    error_p = np.linalg.norm(dp-p)
    print("norm of error of velocity is: ",error_u)
    print("norm of error of pressure is: ",error_p)
    return

if __name__ == '__main__':
    main()