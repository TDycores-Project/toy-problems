include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

ACquad: ACquad.o
	-${CLINKER} -o ACquad ACquad.o ${PETSC_LIB}
	${RM} ACquad.o

distclean:
	@rm -f *~ ACquad *tmp
