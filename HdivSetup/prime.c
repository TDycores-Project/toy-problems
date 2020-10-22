#include <petsc.h>

/*
  This basis was constructed by (1) starting with the hexahedral basis
  found in (Wheeler,Xue,Yotov 2012), but omitting the addition of
  curl(0,0,x*y^2) and curl(0,0,x^2*y*z) to the BDDF1 space, (2)
  creating a constraint matrix to remove x*y from the top and bottom
  faces of the prism and higher order terms from the diagonal face (3)
  taking the SVD and using the eigenvectors to constrain the original
  space.

  p in [0, npoints), i in [0, pdim), c in [0, Nc)

  B[p][i][c] = B[p][i_scalar][c][c]

*/
static PetscErrorCode PetscSpaceEvaluate_HdivPrism(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscInt p, dim = sp->Nv;
  PetscReal x, y, z;
  PetscFunctionBegin;
  if (D || H) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Derivatives and Hessians not supported for Hdiv spaces on prisms");
  }
  if (dim != 3) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Prisms only defined for spatial dimension = 3");
  }
  if (sp->Nc != 3) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Hdiv prism space only defined for number of components = 3");
  }  
  for (p = 0; p < npoints; p++) {
    x = points[p*dim];
    y = points[p*dim+1];
    z = points[p*dim+2];
    B[54*p+ 1] =  1;
    B[54*p+ 4] =  x;
    B[54*p+ 7] =  y;
    B[54*p+10] =  z;
    B[54*p+14] =  1;
    B[54*p+20] =  y;
    B[54*p+23] =  z;
    B[54*p+24] =  x*z;
    B[54*p+25] = -y*z;
    B[54*p+28] = -2*x*z;
    B[54*p+30] =  0.333333333333333*x*x - 0.333333333333333*x*y + 0.577350269189626*y;
    B[54*p+31] =  0.666666666666667*x*y;
    B[54*p+32] = -1.33333333333333*x*z + 0.333333333333333*y*z;
    B[54*p+34] =  2*y*z;
    B[54*p+35] = -z*z;
    B[54*p+36] = -0.707106781186547*x + 0.707106781186547;
    B[54*p+39] =  0.707106781186547*x + 0.707106781186547;
    B[54*p+42] = -0.333333333333333*x*x - 0.666666666666667*x*y - 0.577350269189626*y;
    B[54*p+43] =  0.333333333333333*x*y;
    B[54*p+44] =  0.333333333333333*x*z + 0.666666666666667*y*z;
    B[54*p+45] = -0.666666666666667*x*x - 0.333333333333333*x*y + 0.577350269189626*y;
    B[54*p+46] = -0.333333333333333*x*y;
    B[54*p+47] =  1.666666666666667*x*z + 0.333333333333333*y*z;
    B[54*p+48] = -2*y*z;
    B[54*p+51] =  z; 
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
