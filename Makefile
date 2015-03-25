OBJS = Geometry.o

CXX = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

LIBRARIES = $(OBJS)

.PHONY : all
all: $(LIBRARIES)

Geometry.o: Geometry.hpp Geometry.cpp
	$(CXX) -c $(CFLAGS) $(LIBS) -o Geometry.o Geometry.cpp

.PHONY : clean
clean:
	rm -f $(OBJS)
