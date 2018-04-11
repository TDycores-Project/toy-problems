static char help[] = "";
#include <petsc.h>
#include <petscblaslapack.h>

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
  BLASgemm_("T","N",&r,&r,&q,&one,B,&r,AinvB,&q,&zero,&C[0],&r);

  ierr = PetscFree(pivots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WheelerYotovSystem"
PetscErrorCode WheelerYotovSystem(DM dm)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt v,vStart,vEnd; // bounds of depth  = 0
  PetscInt   fStart,fEnd; // bounds of height = 1
  PetscInt   cStart,cEnd; // bounds of height = 0
  ierr = DMPlexGetDepthStratum (dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  /* Loop over each vertex in the mesh. We will be setting up a local
     problem (B.T * A^-1 * B) which is relative to this vertex. The
     solution of this problem is a stencil which we assemble into the
     global system for pressure. */
  for(v=vStart;v<vEnd;v++){

    if (v != 6) continue; // temporary, just so we look only at the middle vertex

    /* The square matrix A comes from the LHS of (2.43) and is of size
       of the number of faces connected to the vertex. The matrix B
       comes from the RHS of (2.43) and is of size (number of
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

	/* Assemble contributions into B as per (2.51). The 1/2 and
	   |e1| terms will be moved/canceled into A. */
	if (s==0){
	  B[col*nA+row] =  1; // ij --> [j*nrow + i]
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

	    /* Assemble here:
	       Ehat: the area of cell 'support[s]' in the reference domain.
	       s   : 3 for the unit triangle, 4 for the unit square or tetrahedron (below (2.35))
	       Kinv: the inverse permeability tensor pulled back via PK transform, [0,0] component
		     if 'closure[cl] == cone[c]' and [0,1] otherwise
	       E2  : area of 'closure[cl]'
	     */

	    PetscReal Ehat=1,s=1,Kinv=1,E2=1;
	    A[col*nA+row] += 2*Ehat/s*Kinv*E2;
	  }
	}
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&closureSize,&closure);CHKERRQ(ierr);

    /* C = (B.T * A^-1 * B) */
    ierr = FormStencil(&A[0],&B[0],&C[0],nA,nB);CHKERRQ(ierr);


  }
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
  DM        dm,dmDist;
  ierr = DMPlexCreateExodusFromFile(comm,filename,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  // Tell the DM how degrees of freedom interact
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  // Distribute the mesh
  ierr = DMPlexInterpolate(dm,&dmDist);CHKERRQ(ierr);
  if (dmDist) { ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist; }

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
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);

  // Create a vec for the initial condition (constant pressure)
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"RE.");CHKERRQ(ierr);

  ierr = WheelerYotovSystem(dm);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
