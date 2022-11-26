// Separated as openvr.h and openvr_driver.h can't coexist
//
//

#include <openvr.h>
#include <iostream>

// TODO: this should be the whole 'external tracking' handling class - poll for
// tracking data and keep a roundrobin track, allow retrieving arbitrarily for
// registered classes and timestamps
// TODO: this is a bit slapdash as we're separating it away from its native spot
// replace with something more consistent
vr::HmdMatrix34_t getDevicePose(const char * path)
{
    float seconds_since_last_vsync; uint64_t frame_counter;
    vr::VRSystem()->GetTimeSinceLastVsync(&seconds_since_last_vsync, &frame_counter);
    vr::TrackedDevicePose_t devicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::VRSystem()->GetDeviceToAbsoluteTrackingPose(
        vr::TrackingUniverseStanding,
        seconds_since_last_vsync,
        devicePoses,
        vr::k_unMaxTrackedDeviceCount );

    // TODO: get 1s, 1frame, and 0 frames ago HMD, wrists poses
    // TODO: correctly get prior frames
    vr::VRInputValueHandle_t inputHandle = 0;
    auto error2 = vr::VRInput()->GetInputSourceHandle( path,
            &inputHandle);
    if ( error2 != vr::VRInputError_None )
    {
        std::cout << "failed to get input handle? inactive?";
        return vr::HmdMatrix34_t();
    }
    vr::InputOriginInfo_t deviceInfo;
    // Populate deviceInfo with some data about the corresponding device, including deviceIndex
    vr::VRInput()->GetOriginTrackedDeviceInfo(inputHandle, &deviceInfo, sizeof(deviceInfo));

    vr::TrackedDevicePose_t* movePose = devicePoses + deviceInfo.trackedDeviceIndex;

    if (movePose->bPoseIsValid)
    {
        return movePose->mDeviceToAbsoluteTracking;
    }
    return vr::HmdMatrix34_t();
}
