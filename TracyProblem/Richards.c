static char help[] = "Richards equation, constant density\n";
#include <petsc.h>

PETSC_STATIC_INLINE void Waxpy(PetscInt dim, PetscScalar a, const PetscScalar *x, const PetscScalar *y, PetscScalar *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}
PETSC_STATIC_INLINE PetscScalar Dot(PetscInt dim, const PetscScalar *x, const PetscScalar *y) {PetscScalar sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}
PETSC_STATIC_INLINE PetscReal Norm(PetscInt dim, const PetscScalar *x) {return PetscSqrtReal(PetscAbsScalar(Dot(dim,x,x)));}

typedef struct {
  PetscReal *X0,*V0;     // 0-height centroids and volumes
  PetscReal *X1,*V1,*N1; // 1-height centroids, volumes, and normals
} MeshInfo;

typedef struct {
  PetscReal  porosity,n,m,alpha,Sr,Ss,rho,mu,Ki;
  PetscReal *S;    // saturation
  PetscReal *S_Pl; // d/dPl(S)
  PetscReal *Kr;   // relative permeability
  PetscReal *Psi;  // total potential
} SoilState;

typedef struct {
  PetscInt   dim,averaging;
  PetscReal  Pr,g[3],mass;
  SoilState  state;
  MeshInfo   info;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "RelativePermeability_Irmay"
void RelativePermeability_Irmay(PetscReal m,PetscReal Se,PetscReal *Kr,PetscReal *dKr_dSe)
{
  *Kr = PetscPowReal(Se,m);
  if(dKr_dSe) *dKr_dSe = 0;
}

#undef __FUNCT__
#define __FUNCT__ "PressureSaturation_Gardner"
void PressureSaturation_Gardner(PetscReal n,PetscReal m,PetscReal alpha,PetscReal Pc,PetscReal *Se,PetscReal *dSe_dPc)
{
  Pc  = PetscMax(Pc,0);
  *Se = PetscExpReal(-alpha*Pc/m);
  if(dSe_dPc) *dSe_dPc = -alpha/m*PetscExpReal(-alpha*Pc/m);
}

#undef __FUNCT__
#define __FUNCT__ "Solution_Tracy2DSpecifiedHead"
PetscReal Solution_Tracy2DSpecifiedHead(PetscReal *x,AppCtx *user)
{
  SoilState *state  = &(user->state);
  PetscReal  alpha  = state->alpha*(PetscAbsReal(user->g[1])*state->rho); // [1/Pa] --> [1/m]
  PetscReal  L      = 15.24, hr = -L, a = L;                              // assumes the input mesh corresponds, [0,a] X [0,L]
  PetscReal  hbar0  = 1-PetscExpReal(alpha*hr);
  PetscReal  Beta   = PetscSqrtReal(0.25*PetscSqr(alpha)+PetscSqr(PETSC_PI/a));
  PetscReal  hbarss = hbar0*PetscSinReal(PETSC_PI*x[0]/a)*PetscExpReal(0.5*alpha*(L-x[1]))*PetscSinhReal(Beta*x[1])/PetscSinhReal(Beta*L);
  PetscReal  hss    = PetscLogReal(PetscExpReal(alpha*hr)+hbarss)/alpha;
  return user->Pr+state->rho*9.81*hss;
}

#undef __FUNCT__
#define __FUNCT__ "MeshInfoCreate"
PetscErrorCode MeshInfoCreate(DM dm,MeshInfo *info)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt       dim,i,p,pStart,pEnd,nCells;
  PetscReal      dummy[3];
  ierr   = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr   = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  nCells = pEnd-pStart;
  ierr   = PetscMalloc2(nCells*dim*sizeof(PetscReal),&(info->X0),
			nCells    *sizeof(PetscReal),&(info->V0));CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    i    = p-pStart;
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&(info->V0[i]),&(info->X0[i*dim]),dummy);CHKERRQ(ierr);
  }
  ierr   = DMPlexGetHeightStratum(dm,1,&pStart,&pEnd);CHKERRQ(ierr);
  nCells = pEnd-pStart;
  ierr   = PetscMalloc3(nCells*dim*sizeof(PetscReal),&(info->X1),
			nCells*dim*sizeof(PetscReal),&(info->N1),
			nCells*    sizeof(PetscReal),&(info->V1));CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    i    = p-pStart;
    ierr = DMPlexComputeCellGeometryFVM(dm,p,&(info->V1[i]),&(info->X1[i*dim]),&(info->N1[i*dim]));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshInfoDestroy"
PetscErrorCode MeshInfoDestroy(MeshInfo *info)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscFree2(info->X0,info->V0);CHKERRQ(ierr);
  ierr = PetscFree3(info->X1,info->V1,info->N1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SoilStateCreate"
PetscErrorCode SoilStateCreate(DM dm,SoilState *state)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt pStart,pEnd,nCells;
  ierr   = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  nCells = pEnd-pStart;
  ierr   = PetscMalloc4(nCells*sizeof(PetscReal),&(state->S),
			nCells*sizeof(PetscReal),&(state->S_Pl),
			nCells*sizeof(PetscReal),&(state->Kr),
			nCells*sizeof(PetscReal),&(state->Psi));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SoilStateDestroy"
PetscErrorCode SoilStateDestroy(SoilState *state)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = PetscFree4(state->S,state->S_Pl,state->Kr,state->Psi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SoilStateEvaluate"
PetscErrorCode SoilStateEvaluate(DM dm,PetscScalar *Pl,SoilState *state,AppCtx *user)
{
  PetscFunctionBegin;
  PetscErrorCode  ierr;
  PetscInt        dim,i,pStart,pEnd,nCells;
  PetscReal       Se,Se_Pc;
  MeshInfo       *info = &(user->info);
  ierr   = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  nCells = pEnd-pStart;
  dim    = user->dim;
  for(i=0;i<nCells;i++){
    PressureSaturation_Gardner(state->n,state->m,state->alpha,user->Pr-Pl[i],&Se,&Se_Pc);
    RelativePermeability_Irmay(state->m,Se,&(state->Kr[i]),NULL);
    state->S[i]    =  (state->Ss-state->Sr)*Se+state->Sr;
    state->S_Pl[i] = -Se_Pc/(state->Ss-state->Sr);
    state->Psi[i]  =  Pl[i]-state->rho*Dot(dim,user->g,&(info->X0[i*dim]));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RichardsResidual"
PetscErrorCode RichardsResidual(TS ts,PetscReal t,Vec U,Vec U_t,Vec R,void *ctx)
{
  PetscFunctionBegin;
  AppCtx        *user  = (AppCtx *)ctx;
  SoilState     *state = &(user->state);
  MeshInfo      *info  = &(user->info);
  PetscErrorCode ierr;
  DM             dm;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  // Global --> Local of the solution and its time derivative
  Vec          Ul,Ul_t;
  PetscScalar *Pl,*Pl_t;
  ierr = DMGetLocalVector(dm,&Ul  );CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Ul_t);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U  ,INSERT_VALUES,Ul  );CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dm,U  ,INSERT_VALUES,Ul  );CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,U_t,INSERT_VALUES,Ul_t);CHKERRQ(ierr); // don't need ghost values for U_t
  ierr = DMGlobalToLocalEnd  (dm,U_t,INSERT_VALUES,Ul_t);CHKERRQ(ierr);
  ierr = VecGetArray(Ul  ,&Pl  );CHKERRQ(ierr);
  ierr = VecGetArray(Ul_t,&Pl_t);CHKERRQ(ierr);

  // Evaluate equations of state
  ierr = SoilStateEvaluate(dm,Pl,state,user);CHKERRQ(ierr);

  // Compute the residual
  PetscInt     p,pStart,pEnd,ss,dim = user->dim;
  PetscReal   *Rarray,*r,pnt2pnt[3],dist,DarcyFlux,Kr;
  ierr = VecGetArray(R,&Rarray);CHKERRQ(ierr);

  // Residual = d/dt( porosity * S ) ...
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    ierr = DMPlexPointGlobalRef(dm,p,Rarray,&r);CHKERRQ(ierr);
    if(r) *r = state->porosity*state->S_Pl[p]*Pl_t[p];
  }

  // ... - sum( Darcy velocity * interface area / cell volume )
  ierr = DMPlexGetHeightStratum(dm,1,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){

    // Get support of interface
    const PetscInt *supp;
    ierr = DMPlexGetSupportSize(dm,p,&ss  );CHKERRQ(ierr);
    ierr = DMPlexGetSupport    (dm,p,&supp);CHKERRQ(ierr);

    // Handle boundary conditions
    if(ss==1){
      PetscReal *X0 = &(info->X0[supp[0]*dim]); 
      PetscReal *X1 = &(info->X1[(p-pStart)*dim]); 
      Waxpy(dim,-1,X0,X1,pnt2pnt); dist = Norm(dim,pnt2pnt);
      PetscReal  DirichletPl  = Solution_Tracy2DSpecifiedHead(X1,user);
      PetscReal  DirichletPsi = DirichletPl - state->rho*Dot(dim,user->g,X1);
      DarcyFlux = -state->Ki*state->Kr[supp[0]]/state->mu*(DirichletPsi-state->Psi[supp[0]])/dist;
      ierr = DMPlexPointGlobalRef(dm,supp[0],Rarray,&r);CHKERRQ(ierr);
      if(r) *r += DarcyFlux*info->V1[p-pStart]/info->V0[supp[0]];
      continue;
    }

    // Estimate the Darcy flux using a two point flux
    Waxpy(dim,-1,&(info->X0[supp[0]*dim]),&(info->X0[supp[1]*dim]),pnt2pnt); dist = Norm(dim,pnt2pnt);
    if(user->averaging == 0){  // upwind the relative permeability
      Kr = (state->Psi[supp[1]] > state->Psi[supp[0]]) ? state->Kr[supp[1]] : state->Kr[supp[0]];
    }else{                     // average the relative permeability
      Kr = 0.5*(state->Kr[supp[1]]+state->Kr[supp[0]]);
    }
    DarcyFlux = -state->Ki*Kr/state->mu*(state->Psi[supp[1]]-state->Psi[supp[0]])/dist;

    // Load into the residual
    ierr = DMPlexPointGlobalRef(dm,supp[0],Rarray,&r);CHKERRQ(ierr);
    if(r) *r += DarcyFlux*info->V1[p-pStart]/info->V0[supp[0]];
    ierr = DMPlexPointGlobalRef(dm,supp[1],Rarray,&r);CHKERRQ(ierr);
    if(r) *r -= DarcyFlux*info->V1[p-pStart]/info->V0[supp[1]];
  }

  // Cleanup
  ierr = VecRestoreArray(Ul  ,&Pl    );CHKERRQ(ierr);
  ierr = VecRestoreArray(Ul_t,&Pl_t  );CHKERRQ(ierr);
  ierr = VecRestoreArray(R   ,&Rarray);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul  );CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Ul_t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSteadyStateError"
PetscErrorCode ComputeSteadyStateError(DM dm,Vec U,void *ctx)
{
  PetscFunctionBegin;
  AppCtx        *user  = (AppCtx *)ctx;
  MeshInfo      *info  = &(user->info);
  PetscErrorCode ierr;
  PetscInt       p,pStart,pEnd;
  PetscReal     *Pl,*pl;
  PetscReal      exact,error,localL2=0,L2=0,localE=0,E=0;

  ierr = VecGetArray(U,&Pl);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&pStart,&pEnd);CHKERRQ(ierr);
  for(p=pStart;p<pEnd;p++){
    ierr = DMPlexPointGlobalRef(dm,p,Pl,&pl);CHKERRQ(ierr);
    if(pl){
      exact    = Solution_Tracy2DSpecifiedHead(&(info->X0[(p-pStart)*user->dim]),user);
      error    = *pl - exact;
      localL2 += PetscSqr(error)*info->V0[p-pStart];
      localE  += PetscSqr(exact)*info->V0[p-pStart];
    }      
  }
  ierr = VecRestoreArray(U,&Pl);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&localL2,&L2,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
  ierr = MPI_Allreduce(&localE ,&E ,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)U));CHKERRQ(ierr);
  L2   = PetscSqrtReal(L2/E);
  PetscPrintf(PETSC_COMM_WORLD,"L2error %.6e\n",L2);
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

  // Problem parameters
  AppCtx user;
  user.state.m        = 1.0;        // [-]      Saturation-Pressure function parameters
  user.state.alpha    = 1.6717e-05; // [1/Pa]   Saturation-Pressure function parameters
  user.state.Sr       = 0.15;       // [-]      residual (minimum possible) saturation
  user.state.Ss       = 1.0;        // [-]      maximum possible saturation
  user.state.rho      = 1000;       // [kg/m^3] density of water
  user.state.mu       = 9.94e-4;    // [Pa s]   viscosity of water
  user.state.Ki       = 9.1e-9;     // [m^2]    intrinsic permeability (not given in Tracy)
  user.state.porosity = 0.25;       // [-]      porosity
  user.Pr             = 101325;     // [Pa]     reference pressure (atmospheric)
  user.averaging      = 0;          // { 0: upwind, 1: arithmetic average }

  // Options
  char filename[PETSC_MAX_PATH_LEN] = "../data/tracy.e";
  ierr = PetscOptionsBegin(comm,NULL,"Options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-avg","0=upwind, 1=arithmetic average","",user.averaging,&(user.averaging),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Create the mesh
  DM        dm,dmDist;
  PetscInt  overlap=1;
  ierr = DMPlexCreateExodusFromFile(comm,filename,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  // Tell the DM how degrees of freedom interact
  ierr = DMPlexSetAdjacencyUseCone(dm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexSetAdjacencyUseClosure(dm,PETSC_FALSE);CHKERRQ(ierr);

  // Distribute the mesh
  ierr = DMPlexDistribute(dm,overlap,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) { ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist; }
  ierr = DMGetDimension(dm,&(user.dim));CHKERRQ(ierr);

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

  // Setup problem info
  user.g[0] = 0; user.g[1] = 0; user.g[2] = 0; user.g[user.dim-1] = -9.81;
  ierr = MeshInfoCreate(dm,&(user.info));CHKERRQ(ierr);
  ierr = SoilStateCreate(dm,&(user.state));CHKERRQ(ierr);

  // Create a vec for the initial condition (constant pressure)
  Vec U;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = VecSet(U,user.Pr-9810.0*15.24);CHKERRQ(ierr); 
  ierr = PetscObjectSetName((PetscObject)U,"RE.");CHKERRQ(ierr);

  // Create time stepping and solve
  TS  ts;
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,RichardsResidual,&user);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,1000000);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  // Report error
  ierr = ComputeSteadyStateError(dm,U,&user);CHKERRQ(ierr);

  // Cleanup
  ierr = SoilStateDestroy(&(user.state));CHKERRQ(ierr);
  ierr = MeshInfoDestroy(&(user.info));CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return(0);
}
