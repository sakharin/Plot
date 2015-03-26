OBJS = Geometry.o

CXX = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

LIBRARIES = $(OBJS)

.PHONY : all
all: $(LIBRARIES) testCases

Geometry.o: Geometry.hpp Geometry.cpp
	$(CXX) -c $(CFLAGS) $(LIBS) -o Geometry.o Geometry.cpp

testCases: Geometry.o testCases.cpp
	$(CXX) -o testCases -I. \
		Geometry.o \
		testCases.cpp $(CFLAGS) $(LIBS)

.PHONY : clean
clean:
	rm -f $(OBJS) testCases
