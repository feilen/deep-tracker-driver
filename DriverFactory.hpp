#pragma once

#include <cstdlib>
#include <memory>

#include <openvr_driver.h>

#include <IVRDriver.hpp>

#ifdef WINDOWS
extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* interface_name, int* return_code);
#endif

namespace DeepTrackerDriver {
    std::shared_ptr<DeepTrackerDriver::IVRDriver> GetDriver();
}
