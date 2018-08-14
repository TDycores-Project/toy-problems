static char help[] = "";
//#define __DEBUG__

/*
Mary F. Wheeler and Ivan Yotov, 'A Multipoint Flux Mixed Finite
Element Method', SIAM J. Numer. Anal., 44(5), 2082–2106. (25 pages),
https://doi.org/10.1137/050638473
*/
#include <petsc.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt  *bc;
  PetscReal *V,*X,*N,*g;
  PetscScalar *K;
  PetscQuadrature q;
} AppCtx;

/* /\* Exact pressure field given in paper in section 5, page 2103 *\/ */
/* PetscReal Pressure(PetscReal x,PetscReal y) */
/* { */
/*   PetscReal val; */
/*   val  = PetscPowReal(1-x,4); */
/*   val += PetscPowReal(1-y,3)*(1-x); */
/*   val += PetscSinReal(1-y)*PetscCosReal(1-x); */
/*   return val; */
/* } */

/* /\* f = (nabla . -user->K grad(p)) *\/ */
/* PetscReal Forcing(PetscReal x,PetscReal y,PetscScalar *K) */
/* { */
/*   // -k11*(12*(-x + 1)**2 + sin(y - 1)*cos(x - 1)) */
/*   // -k12*( 3*(-y + 1)**2 + sin(x - 1)*cos(y - 1)) */
/*   // -k21*( 3*(-y + 1)**2 + sin(x - 1)*cos(y - 1)) */
/*   // -k22*(-3*(-x + 1)*(2*y - 2) + sin(y - 1)*cos(x - 1)) */
/*   PetscReal val; */
/*   val  = -K[0]*(12*PetscPowReal(1-x,2)+PetscSinReal(y-1)*PetscCosReal(x-1)); */
/*   val += -K[1]*( 3*PetscPowReal(1-y,2)+PetscSinReal(x-1)*PetscCosReal(y-1)); */
/*   val += -K[2]*( 3*PetscPowReal(1-y,2)+PetscSinReal(x-1)*PetscCosReal(y-1)); */
/*   val += -K[3]*(-6*(1-x)*(y-1)+PetscSinReal(y-1)*PetscCosReal(x-1)); */
/*   return val; */
/* } */

PetscReal Pressure(PetscReal x,PetscReal y)
{
  PetscReal val;

  // these are not affected by the permeability tensor nor require a
  // nonzero forcing, on cartesian mesh
  //val = 3.14;  // perfect
  //val = (1-x); // O(h)
  //val = (1-y); // O(h)
  //val = (1-x)+(1-y); // O(h)

  // requires nonzero forcing, on cartesian mesh
  val = (1-x)*(1-y); // O(h)
  //val = (1-x)+(1-y)*x*x; // does not converge
  return val;
}

/* f = (nabla . -user->K grad(p)) */
PetscReal Forcing(PetscReal x,PetscReal y,PetscScalar *K)
{
  PetscReal val=0;
  val = -K[1]-K[2]; // p = (1-x)*(1-y)
  //val = 2*K[0]*(y-1) + 2*x*(K[1]+K[2]); // p = (1-x)+(1-y)*x*x;
  return val;
}

/* Norm of the pressure field given in Remark 4.1 */
PetscErrorCode L2Error(DM dm,Vec U,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscScalar *u,L2;
  PetscInt c,cStart,cEnd;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  PetscSection section;
  PetscInt offset;
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  L2 = 0.;
  for(c=cStart;c<cEnd;c++){
    ierr = PetscSectionGetOffset(section,c,&offset);CHKERRQ(ierr);
    L2  += user->V[c]*PetscSqr(u[offset]-Pressure(user->X[(c-cStart)*2],user->X[(c-cStart)*2+1]));
  }
  printf("%e\n",PetscSqrtReal(L2));
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(user->V ));CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(user->g ));CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscInt ),&(user->bc));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(user->X ));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*(pEnd-pStart)*sizeof(PetscReal),&(user->N ));CHKERRQ(ierr);

  // Compute geometry
  for(p=pStart;p<pEnd;p++){
    if((p >= vStart) && (p < vEnd)) continue;
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&(user->V[p]),&(user->X[p*dim]),&(user->N[p*dim]));CHKERRQ(ierr);
  }

  // Globally constant permeability tensor, given in section 5
  ierr = PetscMalloc(4*sizeof(PetscReal),&(user->K));CHKERRQ(ierr);
  user->K[0] = 5;  user->K[2] = 1;
  user->K[1] = 1;  user->K[3] = 2;

  // Populate Dirichlet conditions
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){

    /* initialize */
    user->bc[p-pStart] = 0;
    user->g [p-pStart] = 0;

    /* faces connected to this cell */
    PetscInt c,coneSize;
    const PetscInt *cone;
    ierr = DMPlexGetConeSize(dm,p,&coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,p,&cone    );CHKERRQ(ierr);
    for(c=0;c<coneSize;c++){

      /* how many cells are connected to this face */
      PetscInt supportSize;
      ierr = DMPlexGetSupportSize(dm,cone[c],&supportSize);CHKERRQ(ierr);
      if ((supportSize == 1) && (user->bc[p-pStart] == 0)) {
	/* this is a boundary cell */
	user->bc[p-pStart] = 1;
	user->g [p-pStart] = Pressure(user->X[p*dim  ],
				      user->X[p*dim+1]);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCtxDestroy"
PetscErrorCode AppCtxDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscFree(user->V );CHKERRQ(ierr);
  ierr = PetscFree(user->X );CHKERRQ(ierr);
  ierr = PetscFree(user->g );CHKERRQ(ierr);
  ierr = PetscFree(user->bc);CHKERRQ(ierr);
  ierr = PetscFree(user->K );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Just to help debug */
void PrintMatrix(PetscScalar *A,PetscInt nr,PetscInt nc)
{
  PetscInt i,j;
  printf("[");
  for(i=0;i<nr;i++){
    printf("  [");
    for(j=0;j<nc;j++){
      printf("%+f, ",A[j*nr+i]);
    }
    printf("],");
    if(i<nr-1) printf("\n");
  }
  printf("]\n");
}

#undef __FUNCT__
#define __FUNCT__ "FormStencil"
PetscErrorCode FormStencil(PetscScalar *A,PetscScalar *B,PetscScalar *C,
			   PetscScalar *G,PetscScalar *D,
			   PetscInt qq,PetscInt rr)
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

  PetscBLASInt q,r,o=1,info,*pivots;
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

  // Solve (A^-1 * G) by back-substitution, stored in G
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,G,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (B.T * AinvB) and (B.T * G)
  PetscScalar zero = 0,one = 1;
  BLASgemm_("T","N",&r,&r,&q,&one,B,&q,AinvB,&q,&zero,&C[0],&r); // B.T * AinvB

  // B.T (rxq) * G (q)
  BLASgemm_("T","N",&r,&o,&q,&one,B,&q,G    ,&q,&zero,&D[0],&r); // B.T * G

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Pullback"
PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,PetscScalar *Kappa,PetscScalar J,PetscInt nn)
{
  /*
    Kappa = J DF^-1 K (DF^-1)^T
    returns Kappa^-1 in column major format
   */

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscScalar  zero=0,one=1;
  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n);CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  PetscScalar KDFinvT[n*n],work[n*n];
  BLASgemm_("N","N",&n,&n,&n,&one,K    ,&n,DFinv  ,&n,&zero,KDFinvT  ,&n); // KDFinvT is now column major
  BLASgemm_("T","N",&n,&n,&n,&J  ,DFinv,&n,KDFinvT,&n,&zero,&Kappa[0],&n); // Kappa is column major

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

#ifdef __DEBUG__
    printf("Vertex %2d x=(%.1f %.1f) ---------------------\n",v,user->X[v*dim],user->X[v*dim+1]);
#endif

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

    /* Space for block matrices */
    PetscScalar A[nA*nA],B[nA*nB],C[nB*nB],G[nA],D[nA];
    PetscMemzero(A,sizeof(PetscScalar)*nA*nA);
    PetscMemzero(B,sizeof(PetscScalar)*nA*nB);
    PetscMemzero(C,sizeof(PetscScalar)*nB*nB);
    PetscMemzero(G,sizeof(PetscScalar)*nA   );
    PetscMemzero(D,sizeof(PetscScalar)*nB   );

    /* In order to assemble A and B, we need to have a local
       mapping. */
    PetscInt i=0,j=0,Amap[nA],Bmap[nB],row,col;
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= fStart) && (closure[cl] < fEnd)) {
	Amap[i] = closure[cl];
#ifdef __DEBUG__
	if(v==8){
	  printf("Amap[%d] = %d\n",i,Amap[i]);
	}
#endif
	i += 1;
      }
      if ((closure[cl] >= cStart) && (closure[cl] < cEnd)) {
	Bmap[j] = closure[cl];
#ifdef __DEBUG__
	if(v==8){
	  printf("Bmap[%d] = %d\n",j,Bmap[j]);
	}
#endif
	j += 1;
      }
    }

    /* The rows of matrices A and B are formed by looping over the
       connected faces to the vertex v. So we grab the closure and
       skip all points which fall outside the range of faces. */
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] < fStart) || (closure[cl] >= fEnd)) continue;

#ifdef __DEBUG__
      PetscInt o = closure[cl];
      printf("  Face %2d x=(%.2f %.2f) n=(%.1f %.1f)\n",closure[cl],user->X[o*dim],user->X[o*dim+1],user->N[o*dim],user->N[o*dim+1]);
#endif

      /* Find the row into which information from this face will be
	 assembled. */
      row = -1;
      for(i=0;i<nA;i++){
	if(Amap[i] == closure[cl]) {
	  row = i;
	  break;
	}
      }

      /* To integrate contributions from this face (test function), we
	 need to get the cells connected to this face. */
      PetscInt s,supportSize;
      const PetscInt *support;
      ierr = DMPlexGetSupportSize(dm,closure[cl],&supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport    (dm,closure[cl],&support    );CHKERRQ(ierr);
      for (s=0;s<supportSize;s++){

#ifdef __DEBUG__
	printf("    Cell %2d\n",support[s]);
#endif

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
	   |e1| terms will be moved/canceled into A. We adopt the
	   convention that positive quantities are assembled when the
	   face centroid points away from the cell centroid. */
	PetscReal dir,val = 1;
	dir  = (user->X[support[s]*dim  ]-user->X[closure[cl]*dim  ])*user->N[closure[cl]*dim  ];
	dir += (user->X[support[s]*dim+1]-user->X[closure[cl]*dim+1])*user->N[closure[cl]*dim+1];
	if(dir > 0) val *= -1;
	B[col*nA+row] = -val; // ij --> [j*nrow + i], column major for LAPACK

	/* Assemble Dirichlet boundary conditions into G (2.50) */
	if (supportSize == 1){
	  if (user->bc[support[s]] == 1) G[row] = dir < 0 ? -user->g[support[s]] : user->g[support[s]];
	}

	/* To assemble the columns of A (trial functions) we then need
	   the faces connected to this cell which we can get from the
	   cone, but only if the original vertex v is also connected,
	   which requires the closure. */
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
	      found = PETSC_TRUE;
	      break;
	    }
	  }
	  ierr = DMPlexRestoreTransitiveClosure(dm,cone[c],PETSC_TRUE,&closureSize2,&closure2);CHKERRQ(ierr);
	  if (found) { /* yes the original vertex is in the closure */

#ifdef __DEBUG__
	    printf("      Face %d\n",cone[c]);
#endif

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

	       1) locate which quadrature point on this element maps
	       to the vertex v
	       2) evaluate the Jacobian (DF) and its inverse/determinant
	       3) transform this element's permeability tensor and then invert it
	       4) choose the component of the tensor that is active
	       based on the mapped element outward normals

	    */
	    PetscReal Ehat = 4;    // area of ref element ( [-1,1] x [-1,1] )
	    PetscReal wgt  = 1./4; // 1/s from the paper
	    PetscInt  q,nq = 4;    // in the end won't be a constant
	    PetscScalar vv[dim*nq],DF[dim*dim*nq],DFinv[dim*dim*nq],J[nq];
	    ierr = DMPlexComputeCellGeometryFEM(dm,support[s],user->q,vv,DF,DFinv,J);CHKERRQ(ierr);

	    for(q=0;q<nq;q++){

	      // Is this the right quadrature point? Does the quad
	      // point map to the vertex v?
	      if ((PetscAbsReal(vv[q*dim  ]-user->X[v*dim  ]) > 1e-12) ||
		  (PetscAbsReal(vv[q*dim+1]-user->X[v*dim+1]) > 1e-12)) continue;

	      // Pullback of the permeability tensor
	      PetscScalar Kappa[dim*dim],Kinv;
	      ierr = Pullback(user->K,&DFinv[dim*dim*q],Kappa,J[q],dim);CHKERRQ(ierr);

	      // Physical edge normals (could point into the
	      // element). The velocity at this vertex/edge is the
	      // scalar u value times this vector.
	      PetscScalar n1[dim],n2[dim];
	      n1[0] = user->N[closure[cl]*dim  ]; // corresponds to test function
	      n1[1] = user->N[closure[cl]*dim+1];
	      n2[0] = user->N[cone   [c ]*dim  ]; // corresponds to trial function
	      n2[1] = user->N[cone   [c ]*dim+1];

	      // What are the reference element-outward normals at
	      // this point? They will be the physical edge normals
	      // pulled back, but we might possibly need to flip them
	      // if they point away from the centroid in physical
	      // space. (Could just use the quadrature number to hard
	      // code these instead.)
	      PetscScalar v1[dim],v2[dim],mag,sign=1;

	      // test function in reference element
	      v1[0] = DFinv[dim*dim*q  ]*n1[0] + DFinv[dim*dim*q+1]*n1[1];
	      v1[1] = DFinv[dim*dim*q+2]*n1[0] + DFinv[dim*dim*q+3]*n1[1];
	      mag   = PetscSqrtScalar(v1[0]*v1[0]+v1[1]*v1[1]);
	      v1[0] = v1[0]/mag;
	      v1[1] = v1[1]/mag;
	      if(((user->X[closure[cl]*dim  ]-user->X[support[s]*dim  ])*n1[0]+
		  (user->X[closure[cl]*dim+1]-user->X[support[s]*dim+1])*n1[1])<0){
		v1[0] *= -1; v1[1] *= -1;
	      }

	      // trial function in reference element
	      v2[0] = DFinv[dim*dim*q  ]*n2[0] + DFinv[dim*dim*q+1]*n2[1];
	      v2[1] = DFinv[dim*dim*q+2]*n2[0] + DFinv[dim*dim*q+3]*n2[1];
	      mag   = PetscSqrtScalar(v2[0]*v2[0]+v2[1]*v2[1]);
	      v2[0] = v2[0]/mag;
	      v2[1] = v2[1]/mag;
	      if(((user->X[cone   [c ]*dim  ]-user->X[support[s]*dim  ])*n2[0]+
		  (user->X[cone   [c ]*dim+1]-user->X[support[s]*dim+1])*n2[1])<0){
		v2[0] *= -1; v2[1] *= -1;
		//sign  *= -1;
	      }

	      // Choose the component (Kappa vhat2, vhat1)
	      Kinv  = (Kappa[0]*v2[0] + Kappa[1]*v2[1])*v1[0];
	      Kinv += (Kappa[2]*v2[0] + Kappa[3]*v2[1])*v1[1];
	      Kinv *= sign;

#ifdef __DEBUG__
	      printf("        v1=(%.1f %.1f) v2=(%.1f %.1f)\n",v1[0],v1[1],v2[0],v2[1]);
	      printf("        A[%2d,%2d] = %+.6f\n",row,col,Kinv);
#endif

	      // Load into the local system, the 2 is the 1/2 from (2.49)
	      A[col*nA+row] += 2*Ehat*wgt*Kinv*user->V[closure[cl]];
	      break;
	    }

	  }
	}
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    ierr = FormStencil(&A[0],&B[0],&C[0],&G[0],&D[0],nA,nB);CHKERRQ(ierr);
    ierr = VecSetValues(F,nB,&Bmap[0],            &D[0],ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(K,nB,&Bmap[0],nB,&Bmap[0],&C[0],ADD_VALUES);CHKERRQ(ierr);
  }

  // Add in the forcing, single quadrature point in the center of the cell
  PetscInt c;
  for(c=cStart;c<cEnd;c++){
    PetscScalar f = Forcing(user->X[c*dim],user->X[c*dim+1],user->K);
    ierr = VecSetValue(F,c,-f*user->V[c],ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
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
  PetscInt N = 8;
  PetscReal P = 0;
  char filename[PETSC_MAX_PATH_LEN] = "../data/simple.e";
  ierr = PetscOptionsBegin(comm,NULL,"Options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","set to 0 or 1 to enable perturbing mesh","",P,&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Create the mesh
  DM        dm;
  const PetscInt  faces[2] = {N  ,N};
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
      PetscReal r = ((PetscReal)rand())/((PetscReal)RAND_MAX)*(P/N*PetscPowReal(2,0.5)/3.); // h*sqrt(2)/3
      PetscReal t = ((PetscReal)rand())/((PetscReal)RAND_MAX)*PETSC_PI;
      coords[offset  ] += r*PetscCosReal(t);
      coords[offset+1] += r*PetscSinReal(t);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  AppCtx user;
  ierr = AppCtxCreate(dm,&user);CHKERRQ(ierr);
  ierr = PetscDTWheelerYotovQuadrature(dm,&user);CHKERRQ(ierr);

  // load vertex locations, this messes up -dm_refine
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){
    ierr = PetscSectionGetOffset(coordSection,v,&offset);CHKERRQ(ierr);
    user.X[v*2  ] = coords[offset  ];
    user.X[v*2+1] = coords[offset+1];
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);

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
  ierr = WheelerYotovSystem(dm,K,F,&user);CHKERRQ(ierr);

  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,F,U);CHKERRQ(ierr);

  L2Error(dm,U,&user);

  ierr = AppCtxDestroy(&user);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
