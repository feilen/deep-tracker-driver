// Copyright (C) Chelsea Jaggi, 2022.
//
// Update model with:
// python3 frugally-deep/keras_export/convert_model.py keras_model.h5 fdeep_model.json

#include <fdeep/fdeep.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "deep-tracker-driver.hpp"
#include "get-device-pose.hpp"

DeepTrackerDriver::TrackerDevice::TrackerDevice(std::string serial):
    serial_(serial)
{
}

std::string DeepTrackerDriver::TrackerDevice::GetSerial()
{
    return this->serial_;
}

void DeepTrackerDriver::TrackerDevice::Update()
{
    if (this->device_index_ == vr::k_unTrackedDeviceIndexInvalid)
        return;

    // Check if this device was asked to be identified
    auto events = GetDriver()->GetOpenVREvents();
    for (auto event : events) {
        // Note here, event.trackedDeviceIndex does not necissarily equal this->device_index_, not sure why, but the component handle will match so we can just use that instead
        //if (event.trackedDeviceIndex == this->device_index_) {
        if (event.eventType == vr::EVREventType::VREvent_Input_HapticVibration) {
            if (event.data.hapticVibration.componentHandle == this->haptic_component_) {
                this->did_vibrate_ = true;
            }
        }
        //}
    }

    // Check if we need to keep vibrating
    if (this->did_vibrate_) {
        this->vibrate_anim_state_ += (GetDriver()->GetLastFrameTime().count()/1000.f);
        if (this->vibrate_anim_state_ > 1.0f) {
            this->did_vibrate_ = false;
            this->vibrate_anim_state_ = 0.0f;
        }
    }

    // Setup pose for this frame
    auto pose = IVRDevice::MakeDefaultPose();

    std::vector<float> inputs;
    for(int i = 0; i < 3; i++)
    {
        for(auto inputPath: input_device_paths)
        {
            auto mat = getDevicePose(inputPath);
            std::vector<float> flatmat{
                mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3],
                mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3],
                mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3]
            };
            inputs.insert(inputs.end(), flatmat.begin(), flatmat.end());
        }
    }

    // TODO: make one model.predict() update many trackers
    const int device_offset = 0;

    // format and feed into model.predict()
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(12),
                                           static_cast<std::size_t>(9)),
        inputs)})[0];

    std::cout << fdeep::show_tensor(result) << std::endl;

    // convert result into vecPosition and qRotation
    pose.vecPosition[0] = result.get(fdeep::tensor_pos(device_offset, 3));
    pose.vecPosition[1] = result.get(fdeep::tensor_pos(device_offset, 7));
    pose.vecPosition[2] = result.get(fdeep::tensor_pos(device_offset, 11));

    Eigen::Matrix3f rotMatrix;
    rotMatrix(0,0) = result.get(fdeep::tensor_pos(device_offset, 0));
    rotMatrix(0,1) = result.get(fdeep::tensor_pos(device_offset, 1));
    rotMatrix(0,2) = result.get(fdeep::tensor_pos(device_offset, 2));
    rotMatrix(1,0) = result.get(fdeep::tensor_pos(device_offset, 4));
    rotMatrix(1,1) = result.get(fdeep::tensor_pos(device_offset, 5));
    rotMatrix(1,2) = result.get(fdeep::tensor_pos(device_offset, 6));
    rotMatrix(2,0) = result.get(fdeep::tensor_pos(device_offset, 8));
    rotMatrix(2,1) = result.get(fdeep::tensor_pos(device_offset, 9));
    rotMatrix(2,2) = result.get(fdeep::tensor_pos(device_offset, 10));

    Eigen::Quaternion<float> poseQuat(rotMatrix);
    pose.qRotation.x = poseQuat.x();
    pose.qRotation.y = poseQuat.y();
    pose.qRotation.z = poseQuat.z();
    pose.qRotation.w = poseQuat.w();

    // TODO: Tell OpenVR about our s k e l e t o n
    // Probably not needed for VRChat, as vrc has its own way of deciding what
    // goes where independent of tracker role
    //vr::VRInputComponentHandle_t skeletal_tracking_component;
    //vr::IVRDriverInput::CreateSkeletonComponent(,
    //        output_device_paths[device_offset], nullptr,
    //        vr::EVRSkeletalTrackingLevel::VRSkeletalTracking::Full, nullptr, 0,
    //        &skeletal_tracking_component);
    //vr::IVRDriverInput::UpdateSkeletonComponent(skeletal_tracking_component,

    // Post pose
    GetDriver()->GetDriverHost()->TrackedDevicePoseUpdated(this->device_index_,
            pose, sizeof(vr::DriverPose_t));
    this->last_pose_ = pose;
}

DeviceType DeepTrackerDriver::TrackerDevice::GetDeviceType()
{
    return DeviceType::TRACKER;
}

vr::TrackedDeviceIndex_t DeepTrackerDriver::TrackerDevice::GetDeviceIndex()
{
    return this->device_index_;
}

vr::EVRInitError DeepTrackerDriver::TrackerDevice::Activate(uint32_t unObjectId)
{
    this->device_index_ = unObjectId;

    GetDriver()->Log("Activating tracker " + this->serial_);

    // Get the properties handle
    auto props = GetDriver()->GetProperties()->TrackedDeviceToPropertyContainer(this->device_index_);

    // Setup inputs and outputs
    GetDriver()->GetInput()->CreateHapticComponent(props, "/output/haptic", &this->haptic_component_);

    GetDriver()->GetInput()->CreateBooleanComponent(props, "/input/system/click", &this->system_click_component_);
    GetDriver()->GetInput()->CreateBooleanComponent(props, "/input/system/touch", &this->system_touch_component_);

    // Set some universe ID (Must be 2 or higher)
    GetDriver()->GetProperties()->SetUint64Property(props, vr::Prop_CurrentUniverseId_Uint64, 2);

    // Set up a model "number" (not needed but good to have)
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_ModelNumber_String, "deep-tracker");

    // Opt out of hand selection
    GetDriver()->GetProperties()->SetInt32Property(props, vr::Prop_ControllerRoleHint_Int32, vr::ETrackedControllerRole::TrackedControllerRole_OptOut);

    // Set up a render model path
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String, "vr_controller_05_wireless_b");

    // Set controller profile
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_InputProfilePath_String, "{example}/input/example_tracker_bindings.json");

    // Set the icon
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceReady_String, "{example}/icons/tracker_ready.png");

    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceOff_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceSearching_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceSearchingAlert_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceReadyAlert_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceNotReady_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceStandby_String, "{example}/icons/tracker_not_ready.png");
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_NamedIconPathDeviceAlertLow_String, "{example}/icons/tracker_not_ready.png");

    return vr::EVRInitError::VRInitError_None;
}

void DeepTrackerDriver::TrackerDevice::Deactivate()
{
    this->device_index_ = vr::k_unTrackedDeviceIndexInvalid;
}

void DeepTrackerDriver::TrackerDevice::EnterStandby()
{
}

void* DeepTrackerDriver::TrackerDevice::GetComponent(const char* pchComponentNameAndVersion)
{
    return nullptr;
}

void DeepTrackerDriver::TrackerDevice::DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize)
{
    if (unResponseBufferSize >= 1)
        pchResponseBuffer[0] = 0;
}

vr::DriverPose_t DeepTrackerDriver::TrackerDevice::GetPose()
{
    return last_pose_;
}

int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(12), static_cast<std::size_t>(9)),
        std::vector<float>{ 0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            0.,0.,0.,0.,0.,0.,0.,0.,0.,
                            })});
    std::cout << fdeep::show_tensors(result) << std::endl;
}
