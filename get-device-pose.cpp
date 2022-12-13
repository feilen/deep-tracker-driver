// Separated as openvr.h and openvr_driver.h can't coexist
//
//

#include <openvr.h>
#include <iostream>

void initialize_client()
{
    auto hmdError = vr::VRInitError_None;
    vr::IVRSystem* pVRSystem = vr::VR_Init(&hmdError, vr::VRApplication_Background);
}

// TODO: this should be the whole 'external tracking' handling class - poll for
// tracking data and keep a roundrobin track, allow retrieving arbitrarily for
// registered classes and timestamps
// TODO: this is a bit slapdash as we're separating it away from its native spot
// replace with something more consistent
vr::HmdMatrix34_t getDevicePose(const char* path)
{
    static bool initialized = false;
    if (!initialized)
    {
        initialize_client();
        initialized = true;
    }
    float seconds_since_last_vsync; uint64_t frame_counter;

    vr::VRSystem()->GetTimeSinceLastVsync(&seconds_since_last_vsync, &frame_counter);
    vr::TrackedDevicePose_t devicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::VRSystem()->GetDeviceToAbsoluteTrackingPose(
        vr::TrackingUniverseStanding,
        seconds_since_last_vsync,
        devicePoses,
        vr::k_unMaxTrackedDeviceCount);

    // TODO: get 1s, 1frame, and 0 frames ago HMD, wrists poses
    // TODO: correctly get prior frames
    vr::VRInputValueHandle_t inputHandle = 0;
    auto error2 = vr::VRInput()->GetInputSourceHandle(path,
        &inputHandle);
    if (error2 != vr::VRInputError_None)
    {
        std::cout << "failed to get input handle? inactive?";
        return vr::HmdMatrix34_t();
    }
    vr::InputOriginInfo_t deviceInfo;
    // Populate deviceInfo with some data about the corresponding device, including deviceIndex
    auto error3 = vr::VRInput()->GetOriginTrackedDeviceInfo(inputHandle, &deviceInfo, sizeof(deviceInfo));
    static double wiggle = 0.;
    static double signn = -1;
    wiggle += 0.0001 * signn;
    if (wiggle > 0.2 || wiggle < -0.2) {
        signn *= -1.;
    }
    if (error3 != vr::VRInputError_None)
    {

        std::cout << "failed to get tracked device info?";
        // XXX: for testing
        auto ret = vr::HmdMatrix34_t();
        ret.m[0][0] = 1.;
        ret.m[1][1] = 1.;
        ret.m[2][2] = 1.;
        if (path != "/user/head") {
            ret.m[0][3] = wiggle;
            ret.m[1][3] = -1.;
        }
        return ret;
    }

    vr::TrackedDevicePose_t* movePose = devicePoses + deviceInfo.trackedDeviceIndex;

    if (deviceInfo.trackedDeviceIndex != vr::k_unTrackedDeviceIndexInvalid && movePose->bPoseIsValid)
    {
        return movePose->mDeviceToAbsoluteTracking;
    }
    return vr::HmdMatrix34_t();
}
