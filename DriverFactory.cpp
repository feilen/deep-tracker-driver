#include "DriverFactory.hpp"
#include <thread>
#include <VRDriver.hpp>
#include <sstream>
#include <string.h>
#include <cstring>
#include <logging.hpp>

static std::shared_ptr<DeepTrackerDriver::IVRDriver> driver;

void static_log(std::string log) {
	if (driver) {
		driver->Log(log);
	}
}

void* HmdDriverFactory(const char* interface_name, int* return_code) {
	if (std::strcmp(interface_name, vr::IServerTrackedDeviceProvider_Version) == 0) {
		if (!driver) {
			// Instantiate concrete impl
			driver = std::make_shared<DeepTrackerDriver::VRDriver>();
			DeepTrackerDriver::SetLoggingFunction(&static_log);
		}
		// We always have at least 1 ref to the shared ptr in "driver" so passing out raw pointer is ok
		return driver.get();
	}

	if (return_code)
		*return_code = vr::VRInitError_Init_InterfaceNotFound;

	return nullptr;
}

std::shared_ptr<DeepTrackerDriver::IVRDriver> DeepTrackerDriver::GetDriver() {
	return driver;
}
