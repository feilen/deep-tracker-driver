#pragma once

#include <openvr_capi.h>
#include <functional>

vr::HmdMatrix34_t getDevicePose(const char * path);