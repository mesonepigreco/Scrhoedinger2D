include ${OCCA_DIR}/scripts/Makefile
LDEXTRA=-lconfig++

all: bin/main.x bin/test_laplace.x

bin/main.x: src/main.cpp src/EvolveWFC.o 
	$(compiler) $(compilerFlags) -o bin/main.x $(flags) src/main.cpp src/EvolveWFC.o $(paths) -L${OCCA_DIR}/lib $(linkerFlags) $(LDEXTRA)

bin/test_laplace.x: src/test_laplace.cpp 
	$(compiler) $(compilerFlags) -o bin/test_laplace.x $(flags) src/test_laplace.cpp src/EvolveWFC.o $(paths) -L${OCCA_DIR}/lib $(linkerFlags) $(LDEXTRA)

src/EvolveWFC.o: src/EvolveWFC.cpp src/EvolveWFC.hpp
	$(compiler) $(compilerFlags) -o src/EvolveWFC.o -c src/EvolveWFC.cpp $(paths) 
