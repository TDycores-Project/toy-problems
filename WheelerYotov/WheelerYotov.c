static char help[] = "";

/*
Mary F. Wheeler and Ivan Yotov, 'A Multipoint Flux Mixed Finite
Element Method', SIAM J. Numer. Anal., 44(5), 2082–2106. (25 pages),
https://doi.org/10.1137/050638473
*/
#include <petsc.h>
#include <petscblaslapack.h>

typedef struct {
  PetscReal *V,*X;
  PetscScalar *K;
  PetscQuadrature q;
} AppCtx;

PetscReal Pressure(PetscReal x,PetscReal y)
{
  PetscReal val;
  val  = PetscPowReal(1-x,4);
  val += PetscPowReal(1-y,3)*(1-x);
  val += PetscSinReal(1-y)*PetscCosReal(1-x);
  return val;
}

/*
  There will need to be a quadrature for each element type in the
  mesh. Neither is this dim independent. It could be generalized to
  simply use as locations the vertices of the reference element.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscDTWheelerYotovQuadrature"
PetscErrorCode PetscDTWheelerYotovQuadrature(DM dm,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF,&(user->q));CHKERRQ(ierr);
  PetscInt dim=2,nq=4;
  PetscReal *x,*w;
  ierr = PetscMalloc1(nq*dim,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(nq    ,&w);CHKERRQ(ierr);
  x[0] = -1.0; x[1] = -1.0;
  x[2] =  1.0; x[3] = -1.0;
  x[4] = -1.0; x[5] =  1.0;
  x[6] =  1.0; x[7] =  1.0;
  w[0] = 0.25; w[1] = 0.25; w[2] = 0.25; w[3] = 0.25;
  ierr = PetscQuadratureSetData(user->q,dim,1,nq,x,w);CHKERRQ(ierr);

  /*
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    printf("cell %d:\n",p);
    PetscReal v[dim*nq],DF[dim*dim*nq],DFinv[dim*dim*nq],J[nq];
    ierr = DMPlexComputeCellGeometryFEM(dm,p,q,v,DF,DFinv,J);CHKERRQ(ierr);
    for(i=0;i<nq;i++){
      printf("   x  =  [%+f, %+f]\n   v  =  [%+f, %+f]\n   DF = [[%+f, %+f],\n         [%+f, %+f]]\n   J  = %f\n\n",x[i*dim],x[i*dim+1],v[i*dim],v[i*dim+1],DF[i*dim*dim],DF[i*dim*dim+1],DF[i*dim*dim+2],DF[i*dim*dim+3],J[i]);
    }
  }
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCtxCreate"
PetscErrorCode AppCtxCreate(DM dm,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt       p,pStart,pEnd,dim=2;
  PetscInt         vStart,vEnd;
  PetscReal      dummy[3];
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(user->V));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(user->X));CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    if((p >= vStart) && (p < vEnd)) continue;
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&(user->V[p]),&(user->X[p*dim]),dummy);CHKERRQ(ierr);
  }
  ierr = PetscMalloc(4*sizeof(PetscReal),&(user->K));CHKERRQ(ierr);
  user->K[0] = 5;  user->K[2] = 1;
  user->K[1] = 1;  user->K[3] = 2;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCtxDestroy"
PetscErrorCode AppCtxDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscFree(user->V);CHKERRQ(ierr);
  ierr = PetscFree(user->X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void PrintMatrix(PetscScalar *A,PetscInt nr,PetscInt nc)
{
  PetscInt i,j;
  printf("[");
  for(i=0;i<nr;i++){
    printf("  [");
    for(j=0;j<nc;j++){
      printf("%+f  ",A[j*nr+i]);
    }
    printf("]");
    if(i<nr-1) printf("\n");
  }
  printf("]\n");
}

#undef __FUNCT__
#define __FUNCT__ "FormStencil"
PetscErrorCode FormStencil(PetscScalar *A,PetscScalar *B,PetscScalar *C,PetscInt qq,PetscInt rr)
{
  // Given block matrices of the form:
  //
  //   | A(qxq)   | B(qxr) |
  //   ---------------------
  //   | B.T(rxq) |   0    |
  //
  // return C = B.T A^-1 B

  PetscFunctionBegin;
  PetscErrorCode ierr;
  
  PetscBLASInt q,r,info,*pivots;
  ierr = PetscBLASIntCast(qq,&q);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rr,&r);CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // Copy B because we will need it again
  PetscScalar AinvB[qq*rr];
  ierr = PetscMemcpy(AinvB,B,sizeof(PetscScalar)*(qq*rr));CHKERRQ(ierr);

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Solve (A^-1 * B) by back-substitution, stored in AinvB
  LAPACKgetrs_("N",&q,&r,A,&q,pivots,AinvB,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (B.T * AinvB)
  PetscScalar zero = 0,one = 1;

  BLASgemm_("T","N",&r,&r,&q,&one,B,&q,AinvB,&q,&zero,&C[0],&r);
  
  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Pullback"
PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,PetscScalar *Kappa,PetscScalar J,PetscInt nn)
{
  /*
    Kappa = J^2  DF^-1 DF^-1 K (DF^-1)^T (DF^-1)^T
    returns Kappa^-1
   */

  PetscFunctionBegin;
  PetscErrorCode ierr;
  
  PetscScalar  zero=0,one=1,J2=J*J;
  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n);CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // DFinv is row major, so we multiply the transpose to get
  // DFinv^2. But now DFinv2 is column major. Kappa is also assumed
  // column major but should be symmetric.
  PetscScalar DFinv2[n*n],KDFinv2T[n*n],work[n*n]; 
  BLASgemm_("T","T",&n,&n,&n,&one,DFinv ,&n,DFinv   ,&n,&zero,DFinv2   ,&n);
  BLASgemm_("T","T",&n,&n,&n,&one,K     ,&n,DFinv2  ,&n,&zero,KDFinv2T ,&n);
  BLASgemm_("N","N",&n,&n,&n,&J2 ,DFinv2,&n,KDFinv2T,&n,&zero,&Kappa[0],&n);

  // Find LU factors of Kappa
  LAPACKgetrf_(&n,&n,&Kappa[0],&n,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Find inverse
  LAPACKgetri_(&n,&Kappa[0],&n,pivots,work,&lwork,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"illegal argument value");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"singular matrix");

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WheelerYotovSystem"
PetscErrorCode WheelerYotovSystem(DM dm,Mat K, Vec F,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt v,vStart,vEnd; // bounds of depth  = 0
  PetscInt   fStart,fEnd; // bounds of height = 1
  PetscInt   cStart,cEnd; // bounds of height = 0
  PetscInt dim = 2;
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  /* Loop over each vertex in the mesh. We will be setting up a local
     problem (B.T * A^-1 * B) which is relative to this vertex. The
     solution of this problem is a stencil which we assemble into the
     global system for pressure. */
  for(v=vStart;v<vEnd;v++){

    //if (v != 8) continue; // temporary, just so we look only at the middle vertex

    /* The square matrix A comes from the LHS of (2.41) and is of size
       of the number of faces connected to the vertex. The matrix B
       comes from the RHS of (2.41) and is of size (number of
       connected faces x number of connected cells). If the vertex is
       an interior one, this is equal to the number of faces. */
    PetscInt *closure = NULL;
    PetscInt  closureSize,cl,nA=0,nB=0;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= fStart) && (closure[cl] < fEnd)) nA += 1;
      if ((closure[cl] >= cStart) && (closure[cl] < cEnd)) nB += 1;
    }
    PetscScalar A[nA*nA],B[nA*nB],C[nB*nB];
    PetscMemzero(A,sizeof(PetscScalar)*nA*nA);
    PetscMemzero(B,sizeof(PetscScalar)*nA*nB);
    PetscMemzero(C,sizeof(PetscScalar)*nB*nB);

    /* In order to assemble A and B, we need to have a local
       mapping. */
    PetscInt i=0,j=0,Amap[nA],Bmap[nB],row,col;
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= fStart) && (closure[cl] < fEnd)) {
	Amap[i] = closure[cl];
	i += 1;
      }
      if ((closure[cl] >= cStart) && (closure[cl] < cEnd)) {
	Bmap[j] = closure[cl];
	j += 1;
      }
    }

    /* The matrices A and B are formed by looping over the connected
       faces to the vertex v. */
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] < fStart) || (closure[cl] >= fEnd)) continue;

      /* Find the row into which information from this face will be
	 assembled. */
      row = -1;
      for(i=0;i<nA;i++){
	if(Amap[i] == closure[cl]) {
	  row = i;
	  break;
	}
      }

      /* We need to get the cells connected to this face. */
      PetscInt s,supportSize;
      const PetscInt *support;
      ierr = DMPlexGetSupportSize(dm,closure[cl],&supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport    (dm,closure[cl],&support    );CHKERRQ(ierr);
      for (s=0;s<supportSize;s++){

	/* Find the column into which information from this cell will
	   be assembled into B. */
	col = -1;
	for(i=0;i<nB;i++){
	  if(Bmap[i] == support[s]) {
	    col = i;
	    break;
	  }
	}

	/* Assemble contributions into B as per (2.49). The 1/2 and
	   |e1| terms will be moved/canceled into A. */
	if (s==0){
	  B[col*nA+row] =  1; // ij --> [j*nrow + i], column major for LAPACK
	}else{
	  B[col*nA+row] = -1;
	}

	/* To assemble A we then need the faces connected to this cell
	   which we can get from the cone, but only if the original
	   vertex v is also connected, which requires the closure. */
	PetscInt c,coneSize;
	const PetscInt *cone;
	ierr = DMPlexGetConeSize(dm,support[s],&coneSize);CHKERRQ(ierr);
	ierr = DMPlexGetCone    (dm,support[s],&cone    );CHKERRQ(ierr);
	for (c=0;c<coneSize;c++){
	  PetscInt *closure2 = NULL;
	  PetscInt  closureSize2,cl2;
	  PetscBool found = PETSC_FALSE;
	  ierr = DMPlexGetTransitiveClosure(dm,cone[c],PETSC_TRUE,&closureSize2,&closure2);CHKERRQ(ierr);
	  for (cl2 = 0; cl2 < closureSize2*2; cl2 += 2) {
	    if (closure2[cl2] == v) {
	      found = PETSC_TRUE; /* yes the original vertex is in the closure */
	      break;
	    }
	  }
	  ierr = DMPlexRestoreTransitiveClosure(dm,cone[c],PETSC_TRUE,&closureSize2,&closure2);CHKERRQ(ierr);
	  if (found) {

	    /* This face corresponds to the column in the local
	       system. Find the local index. */
	    col = -1;
	    for(i=0;i<nA;i++){
	      if(Amap[i] == cone[c]) {
		col = i;
		break;
	      }
	    }

	    /* Assemble contributions into A as per (2.48).
	       
	       I am not going to bother making this general at this
	       point as it is most assuredly not how we will do it in
	       the end. We have 

  	         Kappa = J DF^-1 Khat (DF^-1)^T 

	       where

	         Khat = J DF^-1 K (DF^-1)^T 

	       It only appears to be mapped twice. The apparent second
	       mapping is from pulling back vectors q and v in (K^-1
	       q,v). So we need to:
	       
	       1) locate which quadrature point on this element maps
	       to the vertex v, in this case they are all the same so
	       just use the first one.
	       2) evaluate the Jacobian (DF) and its inverse/determinant
	       3) transform this element's permeability tensor and then invert it
               4) if closure[cl] == cone[c], then Kinv = Kappa^-1 [0,0] else Kappa^-1 [0,1]

	     */
	    PetscReal Ehat = 2;     // area of ref element ( [-1,1] x [-1,1] )
	    PetscReal wgt  = 1./4;  // 1/s from the paper
	    PetscInt  nq    = 4;    // again, in the end won't be constants
	    PetscScalar v[dim*nq],DF[dim*dim*nq],DFinv[dim*dim*nq],J[nq];
	    ierr = DMPlexComputeCellGeometryFEM(dm,support[s],user->q,v,DF,DFinv,J);CHKERRQ(ierr);
	    PetscScalar Kappa[dim*dim];
	    ierr = Pullback(user->K,DFinv,Kappa,J[0],2);CHKERRQ(ierr);
	    PetscReal Kinv = (closure[cl] == cone[c]) ? Kappa[0] : Kappa[2];
	    A[col*nA+row] += 2*Ehat*wgt*Kinv*user->V[closure[cl]];
	  }
	}
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    /* C = (B.T * A^-1 * B) */
    ierr = FormStencil(&A[0],&B[0],&C[0],nA,nB);CHKERRQ(ierr);
    ierr = MatSetValues(K,nB,&Bmap[0],nB,&Bmap[0],&C[0],ADD_VALUES);CHKERRQ(ierr);

    /* I don't really know if this is the right way to set the
       boundary conditions, it is just what I would typically do in a
       FEM code. */
    DMLabel  label;
    PetscInt value;
    ierr = DMGetLabelByNum(dm,2,&label);CHKERRQ(ierr); 
    for(i=0;i<nB;i++){
      ierr = DMLabelGetValue(label,Bmap[i],&value);CHKERRQ(ierr);
      if(value == 1){
	printf("[%f %f] %f\n",
	       user->X[Bmap[i]*dim],
	       user->X[Bmap[i]*dim+1],
	       Pressure(user->X[Bmap[i]*dim],user->X[Bmap[i]*dim+1]));

      }
    }
    
  }

  /* */
  
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  // Initialize
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  // Options
  char filename[PETSC_MAX_PATH_LEN] = "../data/simple.e";
  ierr = PetscOptionsBegin(comm,NULL,"Options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Create the mesh
  DM        dm;
  const PetscInt  faces[2] = {8,8};
  const PetscReal lower[2] = {0.0,0.0};
  const PetscReal upper[2] = {1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,2,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);

  // Perturbed interior vertices from figure 5
  DMLabel      label;
  Vec          coordinates;
  PetscSection coordSection;
  PetscScalar *coords;
  PetscInt     v,vStart,vEnd,offset,value;
  ierr = DMGetLabelByNum(dm,2,&label);CHKERRQ(ierr); // this is the 'marker' label which marks boundary entities
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){
    ierr = PetscSectionGetOffset(coordSection,v,&offset);CHKERRQ(ierr);
    ierr = DMLabelGetValue(label,v,&value);CHKERRQ(ierr);
    if(value==-1){
      PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*0.0589; // h*sqrt(2)/3
      PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
      coords[offset  ] += r*PetscCosReal(t);
      coords[offset+1] += r*PetscSinReal(t);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    
  AppCtx user;
  ierr = AppCtxCreate(dm,&user);CHKERRQ(ierr);
  ierr = PetscDTWheelerYotovQuadrature(dm,&user);CHKERRQ(ierr);

  // Setup the section, 1 dof per cell
  PetscSection sec;
  PetscInt     p,pStart,pEnd;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"LiquidPressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    ierr = PetscSectionSetFieldDof(sec,p,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sec,p,1); CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);

  // Tell the DM how degrees of freedom interact
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  Mat K;
  Vec U,F;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateMatrix      (dm,&K);CHKERRQ(ierr);
  ierr = MatViewFromOptions(K, NULL, "-sys_view");CHKERRQ(ierr);
  //ierr = MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  ierr = WheelerYotovSystem(dm,K,F,&user);CHKERRQ(ierr);

  ierr = AppCtxDestroy(&user);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
