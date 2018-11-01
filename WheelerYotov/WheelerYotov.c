static char help[] = "";
//#define __DEBUG__
#define DIM 2
#define MAX_LOCAL_SIZE 50

/*
Mary F. Wheeler and Ivan Yotov, 'A Multipoint Flux Mixed Finite
Element Method', SIAM J. Numer. Anal., 44(5), 2082–2106. (25 pages),
https://doi.org/10.1137/050638473
*/
#include <petsc.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt  *vmap,*emap,exact;
  PetscReal *V,*X,*N;
  PetscScalar *K;
  PetscQuadrature q;
  PetscReal *Alocal,*Flocal,*vel;
} AppCtx;

/* Just to help debug */
void PrintMatrix(PetscScalar *A,PetscInt nr,PetscInt nc,PetscBool row_major)
{
  PetscInt i,j;
  printf("[[");
  for(i=0;i<nr;i++){
    if(i>0) printf(" [");
    for(j=0;j<nc;j++){
      if(row_major){
	printf("%+.14f, ",A[i*nc+j]);
      }else{
	printf("%+.14f, ",A[j*nr+i]);
      }
    }
    printf("]");
    if(i<nr-1) printf(",\n");
  }
  printf("]\n");
}

PetscErrorCode CheckSymmetric(PetscScalar *A,PetscInt n)
{
  PetscInt i,j;
  PetscErrorCode ierr = 0;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(PetscAbsReal(A[i*n+j]-A[j*n+i]) > 1e-12){
	printf("Symmetry Error A[%d,%d] = %f, A[%d,%d] = %f\n",i,j,A[i*n+j],j,i,A[j*n+i]);
	ierr = 64;
      }
    }
  }
  return ierr;
}

PetscReal Pressure(PetscReal x,PetscReal y,AppCtx *user)
{
  PetscReal val;
  if(user->exact == 0){
    val  = 1-x; 
  }else if(user->exact == 1){
    val  = (x-0.5)*(x-0.5);
  }else{
    /* Exact pressure field given in paper in section 5, page 2103 */
    val  = PetscPowReal(1-x,4);
    val += PetscPowReal(1-y,3)*(1-x);
    val += PetscSinReal(1-y)*PetscCosReal(1-x);
  }
  return val;
}

/* f = (nabla . -user->K grad(p)) */
PetscReal Forcing(PetscReal x,PetscReal y,PetscScalar *K,AppCtx *user)
{
  PetscReal val;
  if(user->exact == 0){
    val = 0; 
  }else if(user->exact == 1){
    val = -2*K[0];
  }else{
    /* Exact forcing from pressure field given in paper in section 5, page 2103 */
    val  = -K[0]*(12*PetscPowReal(1-x,2)+PetscSinReal(y-1)*PetscCosReal(x-1));
    val += -K[1]*( 3*PetscPowReal(1-y,2)+PetscSinReal(x-1)*PetscCosReal(y-1));
    val += -K[2]*( 3*PetscPowReal(1-y,2)+PetscSinReal(x-1)*PetscCosReal(y-1));
    val += -K[3]*(-6*(1-x)*(y-1)+PetscSinReal(y-1)*PetscCosReal(x-1));
  }
  return val;
}

/* v = -user->K grad(p) */
void Velocity(PetscReal x,PetscReal y,PetscScalar *K,PetscScalar *vx,PetscScalar *vy,AppCtx *user)
{
  PetscReal u,v; // grad(p)
  if(user->exact == 0){
    u = -1;
    v =  0;
  }else if(user->exact == 1){
    u = 2*x-1;
    v = 0;
  }else{
    u  = -4*PetscPowReal(1-x,3);
    u += -PetscPowReal(1-y,3);
    u += +PetscSinReal(y-1)*PetscSinReal(x-1);
    v  = -3*PetscPowReal(1-y,2)*(1-x);
    v += -PetscCosReal(x-1)*PetscCosReal(y-1);
  }
  (*vx) = -(K[0]*u + K[1]*v);
  (*vy) = -(K[2]*u + K[3]*v);
  return;
}

PetscErrorCode CheckErrorSpatialDistribution(DM dm, Mat K, Vec F, AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt       c,cStart,cEnd;
  Vec            Ue,Fe;
  ierr = VecDuplicate(F,&Fe);CHKERRQ(ierr);
  ierr = VecDuplicate(F,&Ue);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    ierr = VecSetValue(Ue,c,Pressure(user->X[c*DIM],user->X[c*DIM+1],user),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(Ue);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (Ue);CHKERRQ(ierr);
  ierr = MatMult(K,Ue,Fe);CHKERRQ(ierr);
  ierr = VecAXPY(Fe,-1,F);CHKERRQ(ierr);
  ierr = VecAbs(Fe);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Fe);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (Fe);CHKERRQ(ierr);
  ierr = VecDestroy(&Ue);CHKERRQ(ierr);
  ierr = VecDestroy(&Fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Norm of the pressure field given in Remark 4.1 */
PetscErrorCode L2Error(DM dm,Vec U,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscScalar *u,L2p,L2v,L2d;
  PetscSection section;
  PetscInt c,cStart,cEnd,offset;
  PetscInt f,fStart,fEnd;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);

  // Pressure
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  L2p = 0;
  for(c=cStart;c<cEnd;c++){
    ierr = PetscSectionGetOffset(section,c,&offset);CHKERRQ(ierr);
#ifdef __DEBUG__
    printf("cell %2d: p = %.15f, exact = %.15f\n",c,u[offset],Pressure(user->X[(c-cStart)*2],user->X[(c-cStart)*2+1],user));
#endif
    L2p += user->V[c]*PetscSqr(u[offset]-Pressure(user->X[(c-cStart)*2],user->X[(c-cStart)*2+1],user));
  }
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  // Velocity and Divergence
  L2v = 0; L2d = 0;
  PetscScalar div0,div,flux0,flux,sign,vx,vy,vn;
#ifdef __DEBUG__
  for(f=fStart;f<fEnd;f++){
    Velocity(user->X[DIM*f],user->X[DIM*f+1],user->K,&vx,&vy,user);
    vn = vx*user->N[DIM*f] + vy*user->N[DIM*f+1];
    printf("face %3d: vel = %.15f %.15f, exact = %.15f\n",f,user->vel[DIM*(f-fStart)],user->vel[DIM*(f-fStart)+1],vn);
  }
#endif
  for(c=cStart;c<cEnd;c++){
    PetscInt v,i,j,nf;
    const PetscInt *faces;
    ierr = DMPlexGetConeSize(dm,c,&nf   );CHKERRQ(ierr);
    ierr = DMPlexGetCone    (dm,c,&faces);CHKERRQ(ierr);
    div0 = 0; div = 0; 
    for(i=0;i<nf;i++){
      f = faces[i];
      PetscInt nv;
      const PetscInt *verts;
      ierr = DMPlexGetConeSize(dm,f,&nv   );CHKERRQ(ierr);
      ierr = DMPlexGetCone    (dm,f,&verts);CHKERRQ(ierr);
      sign = PetscSign(user->N[DIM*f  ]*(user->X[DIM*f  ]-user->X[DIM*c  ])+
		       user->N[DIM*f+1]*(user->X[DIM*f+1]-user->X[DIM*c+1]));
      flux0 = 0; flux = 0;
      for(j=0;j<nv;j++){
	v = verts[j];
	Velocity(user->X[DIM*v],user->X[DIM*v+1],user->K,&vx,&vy,user);
	vn     = vx*user->N[DIM*f] + vy*user->N[DIM*f+1];
	flux0 += sign*vn                         *0.5*user->V[f];
	flux  += sign*user->vel[DIM*(f-fStart)+j]*0.5*user->V[f]; 
	div0  += flux0;
	div   += flux; 
      }
      L2v += user->V[c]/user->V[f]*PetscSqr(flux-flux0);
    }
    L2d += user->V[c]*PetscSqr(div-div0);
  }
#ifndef __DEBUG__    
  printf("%e %e %e\n",PetscSqrtReal(L2p),PetscSqrtReal(L2v),PetscSqrtReal(L2d));
#endif
  
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
  PetscInt nq=4;
  PetscReal *x,*w;
  ierr = PetscMalloc1(nq*DIM,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(nq    ,&w);CHKERRQ(ierr);
  x[0] = -1.0; x[1] = -1.0;
  x[2] =  1.0; x[3] = -1.0;
  x[4] = -1.0; x[5] =  1.0;
  x[6] =  1.0; x[7] =  1.0;
  w[0] = 0.25; w[1] = 0.25; w[2] = 0.25; w[3] = 0.25;
  ierr = PetscQuadratureSetData(user->q,DIM,1,nq,x,w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCtxCreate"
PetscErrorCode AppCtxCreate(DM dm,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt p,pStart,pEnd;
  PetscInt   vStart,vEnd;
  PetscInt   fStart,fEnd;  
  PetscInt   cStart,cEnd;
  PetscInt nq=4;
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc(    (pEnd-pStart)*sizeof(PetscReal),&(user->V ));CHKERRQ(ierr);
  ierr = PetscMalloc(DIM*(pEnd-pStart)*sizeof(PetscReal),&(user->X ));CHKERRQ(ierr);
  ierr = PetscMalloc(DIM*(pEnd-pStart)*sizeof(PetscReal),&(user->N ));CHKERRQ(ierr);
  ierr = PetscMalloc(DIM*DIM*nq*(cEnd-cStart)*sizeof(PetscReal),&(user->Alocal));CHKERRQ(ierr);
  ierr = PetscMalloc(           (cEnd-cStart)*sizeof(PetscReal),&(user->Flocal));CHKERRQ(ierr);
  ierr = PetscMalloc(    nq*(cEnd-cStart)*sizeof(PetscInt),&(user->vmap));CHKERRQ(ierr);
  ierr = PetscMalloc(DIM*nq*(cEnd-cStart)*sizeof(PetscInt),&(user->emap));CHKERRQ(ierr);
  ierr = PetscMalloc(DIM   *(fEnd-fStart)*sizeof(PetscReal),&(user->vel));CHKERRQ(ierr);

  /* compute geometry */
  for(p=pStart;p<pEnd;p++){
    if((p >= vStart) && (p < vEnd)) continue;
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&(user->V[p]),&(user->X[p*DIM]),&(user->N[p*DIM]));CHKERRQ(ierr);
  }

  /* globally constant permeability tensor, given in section 5 */
  ierr = PetscMalloc(4*sizeof(PetscReal),&(user->K));CHKERRQ(ierr);
  user->K[0] = 5;  user->K[2] = 1;
  user->K[1] = 1;  user->K[3] = 2;
    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCtxCreateMap"
PetscErrorCode AppCtxCreateMap(DM dm,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i,d,c,cStart,cEnd,f,fStart,fEnd,v,vStart,vEnd,q,nq=4;
  PetscInt local_dirs[8] = {2,1, 3,0, 0,3, 1,2};
  PetscScalar x[DIM*nq],DF[DIM*DIM*nq],DFinv[DIM*DIM*nq],J[nq];
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){

    ierr = DMPlexComputeCellGeometryFEM(dm,c,user->q,x,DF,DFinv,J);CHKERRQ(ierr);
    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);

    /* Check if the image of the quadrature point is coincident with
       the vertex, if so we create a map, local_element_vertex -->
       global_vertex_point */
    for(q=0;q<nq;q++){
      for (i=0;i<closureSize*2;i+=2){
	if ((closure[i] >= vStart) && (closure[i] < vEnd)) {
	  if ((PetscAbsReal(x[q*DIM  ]-user->X[closure[i]*DIM  ]) > 1e-12) ||
	      (PetscAbsReal(x[q*DIM+1]-user->X[closure[i]*DIM+1]) > 1e-12)) continue;
	  user->vmap[c*nq+q] = closure[i];
	  break;
	}
      }
    }

    /* We need a map for (local_element_vertex,direction) -->
       global_face_point. To do this, I loop over the vertices of this
       cell and find connected faces. Then I use the local ordering of
       the vertices to determine where the normal of this face
       points. Finally I check if the normal points into the cell. If
       so, then the index is given a negative as a flag later in the
       assembly process. Since the Hasse diagram always begins with
       cells, there isn't a conflict with 0 being a possible point. */
    for(q=0;q<nq;q++){
      for (i=0;i<closureSize*2;i+=2){
	if ((closure[i] >= fStart) && (closure[i] < fEnd)) {
	  PetscInt fclosureSize,*fclosure = NULL;
	  ierr = DMPlexGetTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,&fclosure);CHKERRQ(ierr);
	  for(f=0;f<fclosureSize*2;f+=2){
	    if (fclosure[f] == user->vmap[c*nq+q]){
	      for(v=0;v<fclosureSize*2;v+=2){
		for(d=0;d<DIM;d++){
		  if (fclosure[v] == user->vmap[c*nq+local_dirs[q*DIM+d]]) {
		    user->emap[c*nq*DIM+q*DIM+d] = closure[i];
		    if(( user->N[closure[i]*DIM  ] * (user->X[closure[i]*DIM  ] - user->X[c*DIM  ]) +
			 user->N[closure[i]*DIM+1] * (user->X[closure[i]*DIM+1] - user->X[c*DIM+1]) ) < 0) {
		      user->emap[c*nq*DIM+q*DIM+d] *= -1;
		    }
		    break;
		  }
		}
	      }
	    }
	  }
	  ierr = DMPlexRestoreTransitiveClosure(dm,closure[i],PETSC_TRUE,&fclosureSize,&fclosure);CHKERRQ(ierr);
	}
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }

#ifdef __DEBUG__
  /* print out for visual inspection */
  printf("Check on vmap/emap ------------------------\n");
  for(c=cStart;c<cEnd;c++){
    printf("cell %d:\n",c);
    for(q=0;q<nq;q++){
      printf("  vertex %2d:",user->vmap[c*nq+q]);
      for(d=0;d<DIM;d++){
	printf(" %+2d",user->emap[c*nq*DIM+q*DIM+d]);
      }
      printf("\n");
    }
    printf("\n");
  }
#endif

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
  ierr = PetscFree(user->K );CHKERRQ(ierr);
  ierr = PetscFree(user->Alocal);CHKERRQ(ierr);
  ierr = PetscFree(user->Flocal);CHKERRQ(ierr);
  ierr = PetscFree(user->vmap);CHKERRQ(ierr);
  ierr = PetscFree(user->emap);CHKERRQ(ierr);
  ierr = PetscFree(user->vel );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormStencil"
PetscErrorCode FormStencil(PetscScalar *A,PetscScalar *B,PetscScalar *C,
			   PetscScalar *G,PetscScalar *D,
			   PetscInt qq,PetscInt rr)
{
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return C(rxr) = B.T A^-1 B in col major
  //        D(r  ) = B.T A^-1 G in col major

  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt q,r,o = 1,info,*pivots;
  PetscScalar zero = 0,one = 1;
  ierr = PetscBLASIntCast(qq,&q);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rr,&r);CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // Copy B because we will need it again
  PetscScalar AinvB[qq*rr];
  ierr = PetscMemcpy(AinvB,B,sizeof(PetscScalar)*(qq*rr));CHKERRQ(ierr); // AinvB in col major

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Solve AinvB = (A^-1 * B) by back-substitution
  LAPACKgetrs_("N",&q,&r,A,&q,pivots,AinvB,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Solve G = (A^-1 * G) by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,G,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  // Compute (B.T * AinvB) and (B.T * G)
  BLASgemm_("T","N",&r,&r,&q,&one,B,&q,AinvB,&q,&zero,&C[0],&r); // B.T * AinvB
  BLASgemm_("T","N",&r,&o,&q,&one,B,&q,G    ,&q,&zero,&D[0],&r); // B.T * G

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RecoverVelocity"
PetscErrorCode RecoverVelocity(PetscScalar *A,PetscScalar *F,PetscInt qq)
{
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // return A U = G - B P
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt q,o = 1,info,*pivots;
  ierr = PetscBLASIntCast(qq,&q);CHKERRQ(ierr);
  ierr = PetscMalloc((q+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  // Find A = LU factors of A
  LAPACKgetrf_(&q,&q,A,&q,pivots,&info);
  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  // Solve by back-substitution
  LAPACKgetrs_("N",&q,&o,A,&q,pivots,F,&q,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Pullback"
PetscErrorCode Pullback(PetscScalar *K,PetscScalar *DFinv,PetscScalar *Kappa,PetscScalar J,PetscInt nn)
{
  /*
    K(dxd)     flattened array in row major (but doesn't matter as it is symmetric)
    DFinv(dxd) flattened array in row major format (how PETSc generates it)
    J          det(DF)

    returns Kappa^-1 = ( J DF^-1 K (DF^-1)^T )^-1
   */

  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscScalar  zero=0,one=1;
  PetscBLASInt n,lwork=nn*nn;
  ierr = PetscBLASIntCast(nn,&n);CHKERRQ(ierr);
  PetscBLASInt info,*pivots;
  ierr = PetscMalloc((n+1)*sizeof(PetscBLASInt),&pivots);CHKERRQ(ierr);

  PetscScalar KDFinvT[n*n],work[n*n];
  /* LAPACK wants things in column major, so we need to transpose both
     K and DFinv. However, we are computing K (DF^-1)^T and thus we
     leave DFinv as is. The resulting KDFinvT is in column major
     format. */
  BLASgemm_("T","N",&n,&n,&n,&one,K    ,&n,DFinv  ,&n,&zero,KDFinvT  ,&n);
  /* Here we are computing J * DFinv * KDFinvT. Since DFinv is row
     major and LAPACK wants things column major, we need to transpose
     it. */
  BLASgemm_("T","N",&n,&n,&n,&J  ,DFinv,&n,KDFinvT,&n,&zero,&Kappa[0],&n);

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
#define __FUNCT__ "WYLocalElementCompute"
PetscErrorCode WYLocalElementCompute(DM dm,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt c,cStart,cEnd;
  PetscInt i,q,nq = 4;
  PetscReal wgt   = 0.25; // 1/s from the paper
  PetscReal Ehat  = 4;    // area of ref element ( [-1,1] x [-1,1] )
  PetscReal ehat  = 2;
  PetscScalar x[DIM*nq],DF[DIM*DIM*nq],DFinv[DIM*DIM*nq],J[nq],Kinv[DIM*DIM];

  // using quadrature points as a local numbering, what are the
  // outward normals for each dof at each local vertex?
  PetscScalar n0[8] = {-1, 0,+1, 0,-1, 0,+1, 0};
  PetscScalar n1[8] = { 0,-1, 0,-1, 0,+1, 0,+1};

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(c=cStart;c<cEnd;c++){
    user->Flocal[c] = 0;
    ierr = DMPlexComputeCellGeometryFEM(dm,c,user->q,x,DF,DFinv,J);CHKERRQ(ierr); // DF/DFinv is row major
    for(q=0;q<nq;q++){

      // compute Kappa^-1 which will be in column major format (shouldn't matter as it is symmetric)
      ierr = Pullback(user->K,&DFinv[DIM*DIM*q],Kinv,J[q],DIM);CHKERRQ(ierr);

      // at each vertex, we have a DIM x DIM system which we will store
      i = c*(DIM*DIM*nq)+q*(DIM*DIM);
      user->Alocal[i  ]  = (Kinv[0]*n0[q*DIM] + Kinv[2]*n0[q*DIM+1])*n0[q*DIM  ]; // (K n0, n0)
      user->Alocal[i  ] += (Kinv[1]*n0[q*DIM] + Kinv[3]*n0[q*DIM+1])*n0[q*DIM+1];
      user->Alocal[i  ] *= Ehat/ehat/ehat*wgt;
      user->Alocal[i+1]  = (Kinv[0]*n1[q*DIM] + Kinv[2]*n1[q*DIM+1])*n0[q*DIM  ]; // (K n1, n0)
      user->Alocal[i+1] += (Kinv[1]*n1[q*DIM] + Kinv[3]*n1[q*DIM+1])*n0[q*DIM+1];
      user->Alocal[i+1] *= Ehat/ehat/ehat*wgt;
      user->Alocal[i+2]  = (Kinv[0]*n0[q*DIM] + Kinv[2]*n0[q*DIM+1])*n1[q*DIM  ]; // (K n0, n1)
      user->Alocal[i+2] += (Kinv[1]*n0[q*DIM] + Kinv[3]*n0[q*DIM+1])*n1[q*DIM+1];
      user->Alocal[i+2] *= Ehat/ehat/ehat*wgt;
      user->Alocal[i+3]  = (Kinv[0]*n1[q*DIM] + Kinv[2]*n1[q*DIM+1])*n1[q*DIM  ]; // (K n1, n1)
      user->Alocal[i+3] += (Kinv[1]*n1[q*DIM] + Kinv[3]*n1[q*DIM+1])*n1[q*DIM+1];
      user->Alocal[i+3] *= Ehat/ehat/ehat*wgt;

      // integrate the forcing function using the same quadrature
      user->Flocal[c] += Forcing(x[q*DIM],x[q*DIM+1],user->K,user)*J[q];
    }
  }

#ifdef __DEBUG__
  printf("Check on local matrices ---------------\n");
  for(c=cStart;c<cEnd;c++){
    printf("cell %d:\n",c);
    for(q=0;q<nq;q++){
      printf("  vertex %d\n",q);
      PrintMatrix(&(user->Alocal[c*(DIM*DIM*nq)+q*(DIM*DIM)]),2,2,PETSC_FALSE);
      PrintMatrix(&(DF[DIM*DIM*q]),2,2,PETSC_FALSE);
    }
  }
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WheelerYotovSystem"
PetscErrorCode WheelerYotovSystem(DM dm,Mat K, Vec F,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt element_vertex,nA,nB,q,nq = 4;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],B[MAX_LOCAL_SIZE],C[MAX_LOCAL_SIZE],G[MAX_LOCAL_SIZE],D[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){ // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(B,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(C,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(G,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(D,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);

    for (c=0;c<closureSize*2;c+=2){ // loop connected cells
      if ((closure[c] < cStart) || (closure[c] >= cEnd)) continue;

	// for the cell, which local vertex is this vertex?
	element_vertex = -1;
	for(q=0;q<nq;q++){
	  if(v == user->vmap[closure[c]*nq+q]){
	    element_vertex = q;
	    break;
	  }
	}
	if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	for(element_row=0;element_row<DIM;element_row++){ // which test function, local to the element/vertex
	  global_row = user->emap[closure[c]*nq*DIM+element_vertex*DIM+element_row]; // DMPlex point index of the face
	  sign_row   = PetscSign(global_row);
	  global_row = PetscAbsInt(global_row);
	  local_row  = -1;
	  for(q=0;q<nA;q++){
	    if(Amap[q] == global_row) {
	      local_row = q; // row into block matrix A, local to vertex
	      break;
	    }
	  }if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  local_col  = -1;
	  for(q=0;q<nB;q++){
	    if(Bmap[q] == closure[c]) {
	      local_col = q; // col into block matrix B, local to vertex
	      break;
	    }
	  }if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  // B here is B.T in the paper, assembled in column major
	  B[local_col*nA+local_row] += 0.5*sign_row*user->V[global_row];

	  // boundary conditions
	  PetscInt isbc;
	  ierr = DMPlexGetSupportSize(dm,global_row,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    G[local_row] += 0.5*sign_row*Pressure(user->X[v*DIM],user->X[v*DIM+1],user)*user->V[global_row];
	  }
	  
	  for(element_col=0;element_col<DIM;element_col++){ // which trial function, local to the element/vertex
	    global_col = user->emap[closure[c]*nq*DIM+element_vertex*DIM+element_col]; // DMPlex point index of the face
	    sign_col   = PetscSign(global_col);
	    global_col = PetscAbsInt(global_col);
	    local_col  = -1; // col into block matrix A, local to vertex
	    for(q=0;q<nA;q++){
	      if(Amap[q] == global_col) {
		local_col = q;
		break;
	      }
	    }if(local_col < 0) {
	      printf("Looking for %d in ",global_col);
	      for(q=0;q<nA;q++){ printf("%d ",Amap[q]); }
	      printf("\n");
	      CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
	    }
	    // Assembled col major, but should be symmetric
	    A[local_col*nA+local_row] += user->Alocal[closure[c]    *(DIM*DIM*nq)+
						      element_vertex*(DIM*DIM   )+
						      element_row   *(DIM       )+
						      element_col]*sign_row*sign_col*user->V[global_row]*user->V[global_col];
	  }
	}
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);
#ifdef __DEBUG__
    printf("A,B,G,C,D of vertex %2d\n",v);
    printf("Amap = [ "); for(q=0;q<nA;q++) { printf("%d ",Amap[q]); }; printf("]\n");
    printf("Bmap = [ "); for(q=0;q<nB;q++) { printf("%d ",Bmap[q]); }; printf("]\n");
    PrintMatrix(A,nA,nA,PETSC_FALSE);
    ierr = CheckSymmetric(A,nA);CHKERRQ(ierr);
    PrintMatrix(B,nA,nB,PETSC_FALSE);
    PrintMatrix(G,nA,1 ,PETSC_FALSE);
#endif
    ierr = FormStencil(&A[0],&B[0],&C[0],&G[0],&D[0],nA,nB);CHKERRQ(ierr);
#ifdef __DEBUG__
    ierr = CheckSymmetric(C,nB);CHKERRQ(ierr);
    PrintMatrix(C,nB,nB,PETSC_FALSE);
    PrintMatrix(D,nB, 1,PETSC_FALSE);
#endif
    /* C and D are in column major, but C is always symmetric and D is
       a vector so it should not matter. */
    ierr = VecSetValues(F,nB,&Bmap[0],            &D[0],ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(K,nB,&Bmap[0],nB,&Bmap[0],&C[0],ADD_VALUES);CHKERRQ(ierr);
  }

  /* Integrate in the forcing */
  for(c=cStart;c<cEnd;c++){
    ierr = VecSetValue(F,c,user->Flocal[c],ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WheelerYotovRecoverVelocity"
PetscErrorCode WheelerYotovRecoverVelocity(DM dm,Vec U,AppCtx *user)
{
  // Given block matrices of the form in col major form:
  //
  //   | A(qxq)   | B(qxr) |   | U |   | G(q) |
  //   --------------------- . ----- = --------
  //   | B.T(rxq) |   0    |   | P |   | F(q) |
  //
  // Compute (- B.T A^-1 B ) U = G - B.T A^-1 F
  // A U = G - B P
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt v,vStart,vEnd;
  PetscInt   fStart,fEnd;
  PetscInt c,cStart,cEnd;
  PetscInt element_vertex,nA,nB,q,nq = 4;
  PetscInt element_row,local_row,global_row;
  PetscInt element_col,local_col,global_col;
  PetscScalar A[MAX_LOCAL_SIZE],F[MAX_LOCAL_SIZE],sign_row,sign_col;
  PetscInt Amap[MAX_LOCAL_SIZE],Bmap[MAX_LOCAL_SIZE];
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  PetscSection section;
  PetscInt     offset;
  PetscScalar *u;
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){ // loop vertices

    PetscInt closureSize,*closure = NULL;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    // determine the size and mapping of the vertex-local systems
    nA = 0; nB = 0;
    for (c = 0; c < closureSize*2; c += 2) {
      if ((closure[c] >= fStart) && (closure[c] < fEnd)) { Amap[nA] = closure[c]; nA += 1; }
      if ((closure[c] >= cStart) && (closure[c] < cEnd)) { Bmap[nB] = closure[c]; nB += 1; }
    }
    ierr = PetscMemzero(A,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);
    ierr = PetscMemzero(F,sizeof(PetscScalar)*MAX_LOCAL_SIZE);CHKERRQ(ierr);

    for (c=0;c<closureSize*2;c+=2){ // loop connected cells
      if ((closure[c] < cStart) || (closure[c] >= cEnd)) continue;

	// for the cell, which local vertex is this vertex?
	element_vertex = -1;
	for(q=0;q<nq;q++){
	  if(v == user->vmap[closure[c]*nq+q]){
	    element_vertex = q;
	    break;
	  }
	}
	if(element_vertex < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	for(element_row=0;element_row<DIM;element_row++){ // which test function, local to the element/vertex
	  global_row = user->emap[closure[c]*nq*DIM+element_vertex*DIM+element_row]; // DMPlex point index of the face
	  sign_row   = PetscSign(global_row);
	  global_row = PetscAbsInt(global_row);
	  local_row  = -1;
	  for(q=0;q<nA;q++){
	    if(Amap[q] == global_row) {
	      local_row = q; // row into block matrix A, local to vertex
	      break;
	    }
	  }if(local_row < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  local_col  = -1;
	  for(q=0;q<nB;q++){
	    if(Bmap[q] == closure[c]) {
	      local_col = q; // col into block matrix B, local to vertex
	      break;
	    }
	  }if(local_col < 0) { CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE); }

	  // B P
	  ierr = PetscSectionGetOffset(section,closure[c],&offset);CHKERRQ(ierr);
	  F[local_row] += 0.5*sign_row*u[offset]*user->V[global_row];

	  // boundary conditions
	  PetscInt isbc;
	  ierr = DMPlexGetSupportSize(dm,global_row,&isbc);CHKERRQ(ierr);
	  if(isbc == 1){
	    F[local_row] -= 0.5*sign_row*Pressure(user->X[v*DIM],user->X[v*DIM+1],user)*user->V[global_row];
	  }
	  
	  for(element_col=0;element_col<DIM;element_col++){ // which trial function, local to the element/vertex
	    global_col = user->emap[closure[c]*nq*DIM+element_vertex*DIM+element_col]; // DMPlex point index of the face
	    sign_col   = PetscSign(global_col);
	    global_col = PetscAbsInt(global_col);
	    local_col  = -1; // col into block matrix A, local to vertex
	    for(q=0;q<nA;q++){
	      if(Amap[q] == global_col) {
		local_col = q;
		break;
	      }
	    }if(local_col < 0) {
	      printf("Looking for %d in ",global_col);
	      for(q=0;q<nA;q++){ printf("%d ",Amap[q]); }
	      printf("\n");
	      CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
	    }
	    // Assembled col major, but should be symmetric
	    A[local_col*nA+local_row] += user->Alocal[closure[c]    *(DIM*DIM*nq)+
						      element_vertex*(DIM*DIM   )+
						      element_row   *(DIM       )+
						      element_col]*sign_row*sign_col*user->V[global_row]*user->V[global_col];
	  }
	}
    }
#ifdef __DEBUG__
    printf("Recovery at vertex %3d:\n",v);
    PrintMatrix(A,nA,nA,PETSC_FALSE);
    PrintMatrix(F,1,nA,PETSC_FALSE);
#endif
    ierr = RecoverVelocity(&A[0],&F[0],nA);CHKERRQ(ierr);
#ifdef __DEBUG__
    PrintMatrix(F,1,nA,PETSC_FALSE);
#endif

    // load velocities
    for(q=0;q<nA;q++){
      global_row = Amap[q];
      const PetscInt *cone;
      ierr = DMPlexGetCone(dm,global_row,&cone);CHKERRQ(ierr);
      offset = -1;
      if(cone[0] == v){
	offset = 0;
      }else if(cone[1] == v){
	offset = 1;
      }else{
	CHKERRQ(PETSC_ERR_ARG_OUTOFRANGE);
      }
      user->vel[DIM*(global_row-fStart)+offset] = F[q]; 
    }

    
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);    
  }

  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
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
  size_t            len;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  // Options
  AppCtx    user;
  PetscInt  N = 8;
  PetscReal P = 0;
  user.exact  = 0;
  char filename[PETSC_MAX_PATH_LEN] = "../data/simple.e";
  char exact_soln_filename[PETSC_MAX_PATH_LEN] = "\0";

  ierr = PetscOptionsBegin(comm,NULL,"Options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","Number of elements in 1D","",N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-E","Which exact solution","",user.exact,&(user.exact),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-P","set to 1 to enable perturbing mesh","",P,&P,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_exact_soln","Filename for saving the exact solution","",exact_soln_filename,exact_soln_filename,sizeof(exact_soln_filename),NULL);CHKERRQ(ierr);
  
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

  ierr = PetscDTWheelerYotovQuadrature(dm,&user);CHKERRQ(ierr);
  ierr = AppCtxCreate(dm,&user);CHKERRQ(ierr);

  // load vertex locations
  ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for(v=vStart;v<vEnd;v++){
    ierr = PetscSectionGetOffset(coordSection,v,&offset);CHKERRQ(ierr);
    user.X[v*2  ] = coords[offset  ];
    user.X[v*2+1] = coords[offset+1];
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  ierr = AppCtxCreateMap(dm,&user);CHKERRQ(ierr);

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

  ierr = WYLocalElementCompute(dm,&user);CHKERRQ(ierr);

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


  ierr = CheckErrorSpatialDistribution(dm,K,F,&user);CHKERRQ(ierr);
  ierr = WheelerYotovRecoverVelocity(dm,U,&user);CHKERRQ(ierr);
  ierr = L2Error(dm,U,&user);CHKERRQ(ierr);
  
  // Save exact solution as binary file
  ierr = PetscStrlen(exact_soln_filename, &len);
  if (len){
    PetscInt       c,cStart,cEnd;
    PetscViewer viewer;
    Vec P;
     ierr = DMCreateGlobalVector(dm,&P);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    for(c=cStart;c<cEnd;c++){
      ierr = VecSetValue(P,c,Pressure(user.X[c*DIM],user.X[c*DIM+1],&user),INSERT_VALUES);CHKERRQ(ierr);
    }
     ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,exact_soln_filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(P,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&P);CHKERRQ(ierr);
  }

  ierr = AppCtxDestroy(&user);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
