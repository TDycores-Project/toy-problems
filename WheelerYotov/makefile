TARGETS = WheelerYotov 

all: ${TARGETS}

WheelerYotov: WheelerYotov.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

CreateMat: CreateMat.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

viz:
	./WheelerYotov -dm_view hdf5:sol.h5 -ksp_monitor_solution hdf5:sol.h5::append -P 1 -N 16
	~/src/PETSc/lib/petsc/bin/petsc_gen_xdmf.py sol.h5

clean::
	${RM} ${TARGETS} *.vtu *.vtk *~ *.xmf *.h5


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
