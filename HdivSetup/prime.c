#include <petsc.h>

/*
  Routine is written so you can comment out a basis row to explore
  effects of removing certain enrichments.

  Input:
    x, point at which to evaluate the prime basis
  Output:
    n, number of basis functions
    P, prime basis of size (basis,component) in row-major
 */
void PrimeBasisWheelerXueYotov(PetscReal *x,PetscInt *n,PetscReal *P){
  PetscInt i=0;
  P[i] =    1; P[i+1] =    0; P[i+2] =    0; i+=3;
  P[i] = x[0]; P[i+1] =    0; P[i+2] =    0; i+=3;
  P[i] = x[1]; P[i+1] =    0; P[i+2] =    0; i+=3;
  P[i] = x[2]; P[i+1] =    0; P[i+2] =    0; i+=3;
  P[i] =    0; P[i+1] =    1; P[i+2] =    0; i+=3;
  P[i] =    0; P[i+1] = x[0]; P[i+2] =    0; i+=3;
  P[i] =    0; P[i+1] = x[1]; P[i+2] =    0; i+=3;
  P[i] =    0; P[i+1] = x[2]; P[i+2] =    0; i+=3;
  P[i] =    0; P[i+1] =    0; P[i+2] =    1; i+=3;
  P[i] =    0; P[i+1] =    0; P[i+2] = x[0]; i+=3;
  P[i] =    0; P[i+1] =    0; P[i+2] = x[1]; i+=3;
  P[i] =    0; P[i+1] =    0; P[i+2] = x[2]; i+=3;
  P[i] =         x[0]*x[2]; P[i+1] =        -x[1]*x[2]; P[i+2] =                 0; i+=3; /* curl(0       ,0       ,x*y*z   ) */
  //P[i] =     2*x[0]*x[1]; P[i+1] =        -x[1]*x[1]; P[i+2] =                 0; i+=3; /* curl(0       ,0       ,x*y**2  ) */
  P[i] =                 0; P[i+1] =      -2*x[0]*x[2]; P[i+2] =                 0; i+=3; /* curl(0       ,0       ,x**2*z  ) */
  //P[i] =  x[0]*x[0]*x[2]; P[i+1] = -2*x[0]*x[1]*x[2]; P[i+2] =                 0; i+=3; /* curl(0       ,0       ,x**2*y*z) */
  P[i] =                 0; P[i+1] =         x[0]*x[1]; P[i+2] =        -x[0]*x[2]; i+=3; /* curl(x*y*z   ,0       ,0       ) */
  P[i] =                 0; P[i+1] =       2*x[1]*x[2]; P[i+2] =        -x[2]*x[2]; i+=3; /* curl(y*z**2  ,0       ,0       ) */
  P[i] =                 0; P[i+1] =                 0; P[i+2] =      -2*x[0]*x[1]; i+=3; /* curl(x*y**2  ,0       ,0       ) */
  P[i] =                 0; P[i+1] =    x[0]*x[1]*x[1]; P[i+2] = -2*x[0]*x[1]*x[2]; i+=3; /* curl(x*y**2*z,0       ,0       ) */
  P[i] =        -x[0]*x[1]; P[i+1] =                 0; P[i+2] =         x[1]*x[2]; i+=3; /* curl(0       ,x*y*z   ,0       ) */
  P[i] =        -x[0]*x[0]; P[i+1] =                 0; P[i+2] =       2*x[0]*x[2]; i+=3; /* curl(0       ,x**2*z  ,0       ) */
  P[i] =      -2*x[1]*x[2]; P[i+1] =                 0; P[i+2] =                 0; i+=3; /* curl(0       ,y*z**2  ,0       ) */
  P[i] = -2*x[0]*x[1]*x[2]; P[i+1] =                 0; P[i+2] =    x[1]*x[2]*x[2]; i+=3; /* curl(0       ,x*y*z**2,0       ) */
  (*n) = i/3;
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  PetscReal x[3],P[72];
  PetscInt i,n;
  x[0] = 0.25; x[1] = -0.12; x[2] = 0.9;
  PrimeBasisWheelerXueYotov(x,&n,P);
  for(i=0;i<3*n;i++) printf("P[%2d] = %+e\n",i,P[i]);
  
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
