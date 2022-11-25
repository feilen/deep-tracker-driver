INCLUDE=-I /usr/include/eigen3/ -I frugally-deep/include -I FunctionalPlus/include -I json/include -I openvr/headers
CPP=g++
CFLAGS=-Og -g

all: deep-tracker-driver.cpp
	$(CPP) -o $@ $^ $(CFLAGS) $(INCLUDE)

.PHONY: clean

clean:
	rm -f all
