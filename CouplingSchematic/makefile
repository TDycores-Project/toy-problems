TARGETS = sample 

all: ${TARGETS}

sample: sample.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

clean::
	${RM} ${TARGETS} *.vtu *.vtk *~ *.xmf *.h5

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
