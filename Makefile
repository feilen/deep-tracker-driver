INCLUDE=-I /usr/include/eigen3/ -I frugally-deep/include -I FunctionalPlus/include -I json/include -I openvr/headers -I.
CPP=g++
CFLAGS=-Og -g -Wall -std=c++17 -Wno-unused-variable

deep-tracker-driver.so: deep-tracker-driver.cpp VRDriver.cpp DriverFactory.cpp get-device-pose.cpp
	$(CPP) --shared -fPIC -o $@ $^ $(CFLAGS) $(INCLUDE) -L openvr/lib/linux64 -l openvr_api

all: deep-tracker-driver.so

.PHONY: clean

clean:
	rm -f deep-tracker-driver
