/*
  https://docs.google.com/document/d/1xQ-zgY9-fXRMlLvMdXhevOeD83M64inyITYb-YPVGGA/edit?usp=sharing
 */
#include <petsc.h>

int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);CHKERRQ(ierr);

  /* Create the spatial mesh: this will be a sample setup 1 [m] by 9
     [m] where 0 corresponds to the surface. Cells with centroids
     above > 0 are for discretizing the canopy and those with
     centroids < 0 are for soil. */
  DM dm;
  const PetscInt  faces[2] = {1  ,9    };
  const PetscReal lower[2] = {-0.5,-5.0};
  const PetscReal upper[2] = {+0.5,+4.0}; 
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,2,PETSC_FALSE,faces,lower,upper,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* We will create two plant types in this cell */
  PetscInt  num_plant_types  = 2;
  PetscReal rooting_depth[2] = {3,2};
  PetscReal canopy_height[2] = {4,3};
  
  /* Setup the section: name the fields and set the number of
     components per field. Here I am treating each entity (soil,
     plant1, and plant2) as a separate field. You may want to define
     more fields, such as soil, plant1_xylem, plant1_leaves, etc. */
  PetscSection sec;
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm),&sec);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1+num_plant_types);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"SoilPressure");CHKERRQ(ierr);  
  ierr = PetscSectionSetFieldComponents(sec,0,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,1,"Plant1Pressure");CHKERRQ(ierr);  
  ierr = PetscSectionSetFieldComponents(sec,1,1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,2,"Plant2Pressure");CHKERRQ(ierr);  
  ierr = PetscSectionSetFieldComponents(sec,2,1);CHKERRQ(ierr);

  /* Loop over grid cells and based on the location of the centroid,
     add dofs to the section */
  PetscReal Xc[2];
  PetscInt  f,i,p,pStart,pEnd,ndof;
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){

    /* Get the centroid of the cell */
    ierr = DMPlexComputeCellGeometryFVM(dm,p,NULL,Xc,NULL);CHKERRQ(ierr);

    /* If we are below the surface, add a dof per cell for field 0, SoilPressure */
    if(Xc[1] < 0){
      f    = 0; // field 0 = SoilPressure
      ierr = PetscSectionSetFieldDof(sec,p,0,1);CHKERRQ(ierr);
      ierr = PetscSectionAddDof     (sec,p,  1);CHKERRQ(ierr);
    }
  }

  /* Now add dofs for the below and above ground part of each plant type */
  for(i=0;i<num_plant_types;i++){
    f = i+1; // Plant field 
    for(p=pStart;p<pEnd;p++){
      
      /* Get the centroid of the cell */
      ierr = DMPlexComputeCellGeometryFVM(dm,p,NULL,Xc,NULL);CHKERRQ(ierr);

      if(Xc[1] < 0){ /* below ground */	
	if(Xc[1] > -rooting_depth[i]){
	  ndof = 1 + 1 + 3; // axial roots + lateral root + rhizosphere
	  ierr = PetscSectionSetFieldDof(sec,p,f,ndof);CHKERRQ(ierr);
	  ierr = PetscSectionAddDof     (sec,p,  ndof);CHKERRQ(ierr);
	}
      }else{ /* above ground */
	if(Xc[1] < canopy_height[i]){
	  ndof = 1 + 1; // xylem + leaves 
	  ierr = PetscSectionSetFieldDof(sec,p,f,ndof);CHKERRQ(ierr);
	  ierr = PetscSectionAddDof     (sec,p,  ndof);CHKERRQ(ierr);
	}
      }
    }
  }
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,sec);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(sec, NULL, "-layout_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec);CHKERRQ(ierr);

  /* Tell the DM how degrees of freedom interact. This will be
     overkill for now (i.e. rhizosphere dofs will interact vertically)
     but we can postpone addressing this as more of an
     optimization. */
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_TRUE);CHKERRQ(ierr);

  /* Global system matrix */
  Mat K;
  ierr = DMCreateMatrix(dm,&K);CHKERRQ(ierr);

  /* Now say we want to deal with plant type 1 separately. */
  DM  dm_plant1;
  IS  is_plant1;
  Mat K_plant1;
  const PetscInt fields[1] = {1};
  ierr = DMCreateSubDM(dm,1,fields,&is_plant1,&dm_plant1);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm_plant1,&K_plant1);CHKERRQ(ierr);
  

  
  ierr = MatDestroy(&K_plant1);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_plant1);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
