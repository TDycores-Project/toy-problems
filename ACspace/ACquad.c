static char help[] =
"Solves the p-Helmholtz equation in 2D using Q_1 FEM.  Option prefix -ph_.\n"
"Problem is posed as minimizing this objective functional over W^{1,p}\n"
"for p>1:\n"
"    I[u] = int_Omega (1/p) |grad u|^p + (1/2) u^2 - f u.\n"
"The strong form equation, namely setting the gradient to zero, is a PDE\n"
"    - div( |grad u|^{p-2} grad u ) + u = f\n"
"subject to homogeneous Neumann boundary conditions.  Implements objective\n"
"and gradient (residual) but no Hessian (Jacobian).  Defaults to linear\n"
"problem (p=2) and quadrature degree 2.  Can be run with only an objective\n" "function; use -ph_no_gradient -snes_fd_function.\n\n";

#include <petsc.h>
#include <petscdt.h>

PetscErrorCode BilinearMap(PetscInt Q, PetscReal Coord_E[4][2], PetscReal xhat[], PetscReal yhat[], PetscReal *x, PetscReal *y);
PetscErrorCode GetQuadrature2D(PetscInt Q1d, PetscReal *qx, PetscReal *qy, PetscReal *w);
int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscInt Q1d = 3;
    PetscReal *qx, *qy, *w, *x, *y;
    PetscInt  i;

    ierr = PetscInitialize(&argc,&argv,NULL,help); if (ierr) return ierr;
    PetscMalloc1(Q1d*Q1d,&qx);
    PetscMalloc1(Q1d*Q1d,&qy);
    PetscMalloc1(Q1d*Q1d,&w);
    PetscMalloc1(Q1d*Q1d,&x);
    PetscMalloc1(Q1d*Q1d,&y);
    ierr = GetQuadrature2D(Q1d, qx, qy, w); CHKERRQ(ierr);
    PetscReal Coord_E[4][2] = {{0., 0.},
                               {1., 0.},
                               {0., 1.},
                               {1., 1.}};
    ierr = BilinearMap(Q1d, Coord_E, qx, qy, x, y); CHKERRQ(ierr);
    for (i = 0; i < Q1d*Q1d; i++){
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "x=%.5f ...\n"
        "y = %.5f\n",
        x[i], y[i]); CHKERRQ(ierr);
    }
    
    PetscFree(qx);
    PetscFree(qy);
    PetscFree(w);
    PetscFree(x);
    PetscFree(y);
    return PetscFinalize();
}


PetscErrorCode BilinearMap(PetscInt Q1d, PetscReal Coord_E[4][2], PetscReal xhat[], PetscReal yhat[], PetscReal *x, PetscReal *y) {
    
    PetscInt   i, j;

    for (j = 0; j<Q1d*Q1d; j++) {
        PetscReal xx = 0.0, yy = 0.0;
        PetscReal N[4];
        N[0] = 0.25 * (1 - xhat[j])*(1 - yhat[j]);
        N[1] = 0.25 * (1 + xhat[j])*(1 - yhat[j]);
        N[2] = 0.25 * (1 - xhat[j])*(1 + yhat[j]);
        N[3] = 0.25 * (1 + xhat[j])*(1 + yhat[j]);

            for (i = 0; i<4; i++) {
                xx += N[i] * Coord_E[i][0];  yy += N[i] * Coord_E[i][1];
            }
    
        x[j] = xx;
        y[j] = yy;
    }
    return 0;
}


PetscErrorCode GetQuadrature2D(PetscInt Q1d, PetscReal *qx, PetscReal *qy, PetscReal *w) {
    
    PetscErrorCode ierr;
    PetscInt   i, j;
    PetscReal *q1, *w1;
    PetscMalloc1(Q1d,&q1);
    PetscMalloc1(Q1d,&w1);
    ierr = PetscDTGaussQuadrature(Q1d,-1.,1., q1, w1); CHKERRQ(ierr);
    for (i = 0; i<Q1d; i++) {
        for (j = 0; j<Q1d; j++) {
            qx[i*Q1d + j] = q1[j];
            qy[i*Q1d + j] = q1[i];
            w[i*Q1d + j] = w1[i]*w1[j];
        }
    }
    PetscFree(q1);
    PetscFree(w1);
    return 0;
}