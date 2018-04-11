TARGETS = Richards

all: ${TARGETS}
	${RM} *.vtu

Richards: Richards.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

clean::
	${RM} ${TARGETS} *.vtu *.vtk *~

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
