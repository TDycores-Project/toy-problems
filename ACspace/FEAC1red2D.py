""" how to run
python3 FEAC1red2D.py --nelx 4 --nely 4 --Q 2 --quadmethod 'GAUSS' --mesh 'uniform'
if you run with 
python3 FEAC1red2D.py
uses the default values as
nelx=4
nely=4
Q=2
quadmethod='GAUSS'
mesh='uniform
"""
import FEACSubroutines as FE
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

def main():
    parser = argparse.ArgumentParser(description='Input for FE_subroutines')

    parser.add_argument('--nelx',
                        dest='nelx',
                        type=int,
                        default=4,
                        help='Number of element in x direction')

    parser.add_argument('--nely',
                        dest='nely',
                        type=int,
                        default=4,
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

    parser.add_argument('--MMS',
                        dest='MMS',
                        type=str,
                        default='quartic',
                        help='MMS solution which can be:trig, quartic')

    args = parser.parse_args()

    nelx = args.nelx
    nely = args.nely
    Q = args.Q
    quadmethod = args.quadmethod
    mesh = args.mesh
    MMS = args.MMS

    F, K = FE.Assembly(MMS, mesh, nelx, nely, Q, quadmethod)

    dp, du = FE.GetFESol(K,F,nelx,nely)
    p, u = FE.GetExactSol(MMS, mesh, nelx, nely)
    error_u = np.absolute(du-u)
    error_p = np.absolute(dp-p)
    normerror_u = np.linalg.norm(du-u)
    normerror_p = np.linalg.norm(dp-p)
    print("norm of error of velocity is: ",normerror_u)
    print("norm of error of pressure is: ",normerror_p)
    #for i in range(len(u)):
    #    print(i+1,'---',u[i,0],'====',du[i,0])

    FE.PltSolution(mesh, nelx, nely, u, p)
    FE.PltSolution(mesh, nelx, nely, du, dp)
    FE.PltSolution(mesh, nelx, nely, error_u, error_p)

    return

if __name__ == '__main__':
    main()