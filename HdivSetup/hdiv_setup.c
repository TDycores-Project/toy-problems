#include "petsc.h"

int main(int argc, char **argv)
{
  /* Initialize */
  PetscErrorCode ierr;
  PetscInt N = 4, dim = 2;
  ierr = PetscInitialize(&argc,&argv,(char*)0,0);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Transient Options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N"  ,"Number of elements in 1D","",N  ,&N  ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Spatial dimensions"      ,"",dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  
  /* Create and distribute the mesh */
  DM dm, dmDist = NULL;
  const PetscInt  faces[3] = {N  ,N  ,N  };
  const PetscReal lower[3] = {0.0,0.0,0.0};
  const PetscReal upper[3] = {1.0,1.0,1.0};
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 1, NULL, &dmDist);  
  if (dmDist) {DMDestroy(&dm); dm = dmDist;}
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Get plex limits */
  PetscInt pStart,pEnd,c,cStart,cEnd,f,fStart,fEnd;
  ierr = DMPlexGetChart        (dm,  &pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);

  /* Create H-div section */
  PetscSection sec;
  ierr = PetscSectionCreate(PETSC_COMM_WORLD,&sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,2);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Pressure");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,1,"Velocity");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,1,1);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  
  /* Setup 1 dof per cell for field 0 */
  for(c=cStart;c<cEnd;c++){
    ierr = PetscSectionSetFieldDof(sec,c,0,1); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,c  ,1); CHKERRQ(ierr);
  }

  /* Setup dofs_per_face considering quads and hexes only */
  PetscInt d,dofs_per_face = 1;
  for(d=0;d<(dim-1);d++) dofs_per_face *= 2;
  for(f=fStart;f<fEnd;f++){
    ierr = PetscSectionSetFieldDof(sec,f,1,dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec,f  ,dofs_per_face); CHKERRQ(ierr);
  }
  
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);

  /* I am not sure what we want here, but this seems to be a
     conservative estimate on the sparsity we need. */
  ierr = DMPlexSetAdjacencyUseCone   (dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  Mat A;
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);

  /* Assemble a 1 in the pressure diagonal, would be 0 in a steady run,
     diagonal for transient. */
  PetscInt row,col,junk;
  for(c=cStart;c<cEnd;c++){
    ierr = DMPlexGetPointGlobal(dm,c,&row,&junk);CHKERRQ(ierr);
    if(row < 0) continue; /* we do not own this row */
    ierr = MatSetValue(A,row,row,1,ADD_VALUES);CHKERRQ(ierr);
  }

  /* Assemble a 2 in the off diagonal, each face's dofs interact with
     the neighboring cells. */
  PetscInt dStart,dEnd;
  for(f=fStart;f<fEnd;f++){

    /* what are the columns for this face's dofs? */
    ierr = DMPlexGetPointGlobal(dm,f,&dStart,&dEnd);CHKERRQ(ierr);
    
    /* which cells are in the support */
    PetscInt s,support_size;
    const PetscInt *support;
    ierr = DMPlexGetSupportSize(dm,f,&support_size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport    (dm,f,&support     );CHKERRQ(ierr);
    for(s=0;s<support_size;s++){
      ierr = DMPlexGetPointGlobal(dm,support[s],&row,&junk);CHKERRQ(ierr);
      if(row < 0) continue; /* we do not own this row */
      for(col=dStart;col<dEnd;col++){
	ierr = MatSetValue(A,row,col,2,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Cleanup */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
