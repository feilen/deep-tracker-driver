// Separated as openvr.h and openvr_driver.h can't coexist
//
//

#include <openvr.h>
#include <iostream>
#include <cassert>
#include <logging.hpp>

void initialize_client()
{
    auto hmdError = vr::VRInitError_None;
    vr::IVRSystem* pVRSystem = vr::VR_Init(&hmdError, vr::VRApplication_Utility);
    assert(hmdError == vr::VRInitError_None);
}

// TODO: this should be the whole 'external tracking' handling class - poll for
// tracking data and keep a roundrobin track, allow retrieving arbitrarily for
// registered classes and timestamps
// TODO: this is a bit slapdash as we're separating it away from its native spot
// replace with something more consistent
vr::HmdMatrix34_t getDevicePose(const char* path)
{
    vr::EVRInitError eError;
    void* m_pVRCompositor = (vr::IVRCompositor*)VR_GetGenericInterface(vr::IVRSystem_Version, &eError);
    if (eError != vr::EVRInitError::VRInitError_None)
    {
        initialize_client();
    }
    //float seconds_since_last_vsync; uint64_t frame_counter;

    //vr::VRSystem()->GetTimeSinceLastVsync(&seconds_since_last_vsync, &frame_counter);


    vr::TrackedDevicePose_t devicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::VRSystem()->GetDeviceToAbsoluteTrackingPose(
        vr::TrackingUniverseStanding,
        0.01f,
        devicePoses,
        vr::k_unMaxTrackedDeviceCount);

    // TODO: get 1s, 1frame, and 0 frames ago HMD, wrists poses
    // TODO: correctly get prior frames
    vr::VRInputValueHandle_t inputHandle = 0;
    assert(vr::VRInput());
    const auto inputhdl = vr::VRInput();
    auto error2 = vr::VRInput()->GetInputSourceHandle(path,
        &inputHandle);
    if (error2 != vr::VRInputError_None)
    {
        DeepTrackerDriver::Log("failed to get input handle? inactive?");
        return vr::HmdMatrix34_t();
    }
    //vr::InputOriginInfo_t deviceInfo;
 
    // Populate deviceInfo with some data about the corresponding device, including deviceIndex - it doesn't seem to like /me/hand/left
    /*auto error3 = vr::VRInput()->GetOriginTrackedDeviceInfo(inputHandle, &deviceInfo, sizeof(deviceInfo));
    if (error3 != vr::VRInputError_None)
    {

        DeepTrackerDriver::Log("failed to get tracked device info?");
        return vr::HmdMatrix34_t();
    }
    */
    
    vr::TrackedDeviceIndex_t dev_index;
    if (path == "/user/hand/left") {
        dev_index = vr::VRSystem()->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_LeftHand);
    }
    else if (path == "/user/hand/right") {
        dev_index = vr::VRSystem()->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand);
    }
    else {
        dev_index = vr::k_unTrackedDeviceIndex_Hmd;
    }
    
    vr::TrackedDevicePose_t* movePose = devicePoses + dev_index;
    //vr::TrackedDevicePose_t* movePose = devicePoses + deviceInfo.trackedDeviceIndex;

    if (dev_index != vr::k_unTrackedDeviceIndexInvalid 
        && movePose->eTrackingResult == vr::TrackingResult_Running_OK
        && movePose->bPoseIsValid)
    {
        return movePose->mDeviceToAbsoluteTracking;
    }
    return vr::HmdMatrix34_t();
}
