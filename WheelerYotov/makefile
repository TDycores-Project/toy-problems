TARGETS = WheelerYotov 

all: ${TARGETS}

WheelerYotov: WheelerYotov.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

CreateMat: CreateMat.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

clean::
	${RM} ${TARGETS} *.vtu *.vtk *~

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
