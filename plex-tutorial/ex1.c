
static char help[] ="DMPlex primer";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscds.h>
#include <petscts.h>
#include <petscsf.h> /* For SplitFaces() */
#include <petscfvtypes.h>

int main(int argc,char **argv)
{
  PetscInt       dim, ncells, nverts, cnt;
  PetscInt       p, pStart, pEnd;
  PetscInt       nx[3];
  PetscInt       nverts_per_cell;
  PetscInt       ii,jj,kk;
  PetscInt       *cells;
  PetscInt       vert_ids[4][3][2];
  PetscReal      *verts;
  PetscReal      dx[3];
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       depth, dStart, dEnd, hStart, hEnd;
  PetscInt       ss;
  DM             dm, dmDist;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);


  dim    = 3;

  nx[0] = 2; nx[1] = 1; nx[2] = 1;
  dx[0] = 1.; dx[1] = 2.; dx[2] = 3.;
  
  nverts_per_cell = 8;
  
  ncells = nx[0]*nx[1]*nx[2];
  nverts = (nx[0]+1)*(nx[1]+1)*(nx[2]+1);

  PetscMalloc1(ncells*nverts_per_cell,&cells);
  PetscMalloc1(nverts*3,&verts);

  printf("Vertex coordinates:\n");
  cnt = 0;
  for (kk=0;kk<nx[2]+1;kk++) {
    for (jj=0;jj<nx[1]+1;jj++) {
      for (ii=0;ii<nx[0]+1;ii++) {
        verts[3*cnt + 0] = ii * dx[0];
        verts[3*cnt + 1] = jj * dx[1];
        verts[3*cnt + 2] = kk * dx[2];
        printf("(%d) %f %f %f\n",cnt,verts[3*cnt + 0],verts[3*cnt + 1],verts[3*cnt + 2]);
        vert_ids[ii][jj][kk] = cnt;
        cnt++;
      }
    }
  }
  
  printf("\nVertex connectivity:\n");
  cnt = 0;
  for (kk=0;kk<nx[2];kk++) {
    for (jj=0;jj<nx[1];jj++) {
      for (ii=0;ii<nx[0];ii++) {

        cells[8*cnt+0] = vert_ids[ii  ][jj  ][kk  ];
        cells[8*cnt+3] = vert_ids[ii+1][jj  ][kk  ];
        cells[8*cnt+2] = vert_ids[ii+1][jj+1][kk  ];
        cells[8*cnt+1] = vert_ids[ii  ][jj+1][kk  ];

        cells[8*cnt+4] = vert_ids[ii  ][jj  ][kk+1];
        cells[8*cnt+5] = vert_ids[ii+1][jj  ][kk+1];
        cells[8*cnt+6] = vert_ids[ii+1][jj+1][kk+1];
        cells[8*cnt+7] = vert_ids[ii  ][jj+1][kk+1];

        printf("(%d) %d %d %d %d %d %d %d %d\n",cnt,
        cells[8*cnt+0],cells[8*cnt+1],cells[8*cnt+2],cells[8*cnt+3],
        cells[8*cnt+4],cells[8*cnt+5],cells[8*cnt+6],cells[8*cnt+7]);

        cnt++;
      }
    }
  }

  ierr = DMPlexCreateFromCellList(PETSC_COMM_WORLD,dim,ncells,
  nverts,nverts_per_cell,PETSC_TRUE,cells,dim,verts,&dm);
  
  /*
  ierr = DMPlexCreateHexBoxMesh(PETSC_COMM_WORLD, dim, cells,
    DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, &dm);
    */

  ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);


  ierr = DMPlexSetAdjacencyUseCone(dm, PETSC_TRUE);
  ierr = DMPlexSetAdjacencyUseClosure(dm, PETSC_TRUE);

  ierr = DMPlexDistribute(dm,1,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) { ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist; }
  
  ierr = DMSetFromOptions(dm);

  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMPlexGetChart(dm, &pStart, &pEnd);
  printf("\nCone and support info for all mesh points in DMPlex:\n");
  printf("pStart = %d pEnd = %d\n",pStart,pEnd);
  for (p = pStart; p < pEnd; p++){
    const PetscInt *points;
    PetscInt       size;
    
    DMPlexGetConeSize(dm, p, &size);
    DMPlexGetCone(    dm, p, &points);
    printf("Point = %02d; Cone = %d [ ",p, size);
    for (ii = 0; ii<size; ii++)  printf("%02d ",points[ii]);
    for (ii = 0; ii<6-size;ii++) printf("   ");
    printf("] ");

    DMPlexGetSupportSize(dm, p, &size);
    DMPlexGetSupport(    dm, p, &points);
    printf("Support = [ ");
    for (ii = 0; ii<size; ii++) printf("%02d ",points[ii]);
    for (ii = 0; ii<3-size;ii++) printf("   ");
    printf("]\n");

  }
  
  printf("\nStratum info at various depths:\n");
  DMPlexGetDepth(dm, &depth);
  printf("Maximum depth = %d\n",depth);
  for (ii = 0; ii<=depth; ii++){
    DMPlexGetHeightStratum(dm, ii, &hStart, &hEnd);
    DMPlexGetDepthStratum( dm, ii, &dStart, &dEnd);
    printf("  (depth=%d) height stratum= %02d %02d   depth stratum= %02d %02d\n",ii,hStart,hEnd-1,dStart,dEnd-1);
  }
  
  printf("\nInformation about faces:\n");
  ierr = DMPlexGetHeightStratum(dm,1,&hStart,&hEnd);CHKERRQ(ierr);
  for (ii = hStart; ii<hEnd; ii++) {
    const PetscInt *supp;
    ierr = DMPlexGetSupportSize(dm,ii,&ss  );CHKERRQ(ierr);
    ierr = DMPlexGetSupport    (dm,ii,&supp);CHKERRQ(ierr);
    printf("(%d) num neighbors = %d  IDs of control volue = [",ii,ss);
    for (jj = 0; jj<ss; jj++) printf("%d ",supp[jj]);
    printf("]\n");
  }
  
  /*
  DMPlexGetHeightStratum(dm, 0, &hStart, &hEnd);
  for (ii=hStart; ii<hEnd; ii++){
    PetscReal vol, centroid[3], normal[3];
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&vol,&centroid,&normal);CHKERRQ(ierr);
    printf("(%d) vol = %f\tcentroid = %f %f %f\tnormal = %f %f %f\n",ii,vol,
      centroid[0],centroid[1],centroid[2],normal[0],normal[1],normal[2]);
  }
  */
  /*
    ierr = DMPlexSNESGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);

    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    printf("fStart = %d %d\n",fStart,fEnd);
    ierr = DMPlexGetFaceGeometry(dm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
  */

  ierr = PetscFinalize();
  return ierr;
}




