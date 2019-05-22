from sympy import *

d,nc = symbols('d nc')

nf  = d*nc     # for periodic structured meshes of equal cells in each dimension
ncv = 2**(d  ) # number of cell vertices
nfv = 2**(d-1) # number of face vertices

# For BDM

# each velocity dof interacts with all velocity dofs in both
# neighboring cells, but do not double count the dofs on the shared
# face
per_vdof = 2*d*ncv - nfv
A = per_vdof*nfv*nf

# each pressure dof interacts with itself and all velocity dofs of its
# faces
per_pdof = 1 + ncv*d
B = per_pdof*nc

# total is A block and twice the B block
total_bdm = A + 2*B
print(total_bdm.subs({d:3,nc:5040})) # 2,913,120 vs 3,911,040 of -mat_view

# For WY

# each pressure dof interacts with all its neighbors, even diagonal ones
per_pdof = 3**d
total_wy = per_pdof*nc
print(total_wy.subs({d:3,nc:5040})) # 136,080 vs 118,720 of -mat_view

# ratio
print((total_bdm / total_wy).subs({d:3.})) # 21.4 times more memory

# but for WY I also store dxd systems at each element/vertex
total_wy = (per_pdof + ncv*d**2)*nc
print((total_bdm / total_wy).subs({d:3.})) # 5.8 times more memory
