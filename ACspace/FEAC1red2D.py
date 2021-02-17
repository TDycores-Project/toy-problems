""" how to run
python3 FEAC1red2D.py --nelx 4 --nely 4 --Q 2 --quadmethod GAUSS --mesh uniform --MMS quartic
if you run with 
python3 FEAC1red2D.py
uses the default values as
nelx=4
nely=4
Q=2
quadmethod=GAUSS
mesh=uniform
MMS = quartic
"""
import FEACSubroutines as FE
import argparse
import numpy as np

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
                        help='MMS solution which can be:trig, quartic, quadratic, linear, constant')

    parser.add_argument('--problem',
                        dest='problem',
                        type=str,
                        default='convrate',
                        help='run convergence rate problem')
    args = parser.parse_args()

    nelx = args.nelx
    nely = args.nely
    Q = args.Q
    quadmethod = args.quadmethod
    mesh = args.mesh
    MMS = args.MMS
    problem = args.problem
    edge = 'all'

    if problem == 'convrate':
        F1, K = FE.Assembly(MMS, mesh, nelx, nely, Q, quadmethod)

        T = FE.GetGlobalTraction(MMS, mesh, nelx, nely, Q, quadmethod, edge)
        F = F1 - T

        dp, du = FE.GetFESol(K,F,nelx,nely)
        p, u = FE.GetExactSol(MMS, mesh, nelx, nely)
        h = FE.Gethsz(mesh, nelx, nely)
        #res_p, res_u = FE.GetResidual(K,u,p, nelx, nely)

        #error_u = np.absolute(du-u)
        #error_p = np.absolute(dp-p)
        normerror_u = np.linalg.norm(du-u)/np.linalg.norm(u)
        normerror_p = np.linalg.norm(dp-p)/np.linalg.norm(p)
        
        print("h, Velocity and Pressure Absolute Error:",h, normerror_u, normerror_p)
    
        FE.plotmesh(mesh, nelx, nely)
        FE.PltSolution(mesh, nelx, nely, du, dp, 'ux_h','uy_h','p_h' )
        #FE.PltSolution(mesh, nelx, nely, u, p,'ux_ex','uy_ex','p_ex')
        #FE.PltSolution(mesh, nelx, nely, error_u, error_p,'abs(ux_ex - ux_h)', 'abs(uy_ex - uy_h)', 'abs(p_ex - p_h)')
        #FE.PltSolution(mesh, nelx, nely, res_u, res_p, 'res_ux','res_uy','res_p')

    if problem == 'infsup':
        H, M, B, C = FE.GetGlobalInfSupMat(mesh, nelx, nely, Q, quadmethod)
        alpha, beta = FE.GetInfSupConst(H, M, B, C)
        h = FE.Gethsz(mesh, nelx, nely)
        FE.plotmesh(mesh, nelx, nely)
        print("h, coercivity and infsup constants:",h, alpha, beta)

    return

if __name__ == '__main__':
    main()