TARGETS = hdiv_setup

all: ${TARGETS}

hdiv_setup: hdiv_setup.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

sample: hdiv_setup
	@mpirun -n 2 ./hdiv_setup -N 4 -dim 2 -mat_view ascii:mat.dat
	@python plot_mat.py mat.dat

clean::
	${RM} ${TARGETS} *~ *.dat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
