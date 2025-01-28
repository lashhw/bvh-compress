all: FORCE
	g++ -Wall -g main.cpp -Ibvh/include -fopenmp
clean:
	rm a.out
FORCE:
