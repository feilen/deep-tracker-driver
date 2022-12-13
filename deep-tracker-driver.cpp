// Copyright (C) Chelsea Jaggi, 2022.
//
// Update model with:
// python3 frugally-deep/keras_export/convert_model.py keras_model.h5 fdeep_model.json

#include <Eigen/Geometry>
#include <cmath>

#include "deep-tracker-driver.hpp"
#include "get-device-pose.hpp"

DeepTrackerDriver::TrackerDevice::TrackerDevice(std::string serial, int device_offset) :
    serial_(serial),
    prediction_model(fdeep::load_model("C:/fdeep_model.json")),
    device_offset(device_offset)
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

    // Setup pose for this frame
    auto pose = IVRDevice::MakeDefaultPose();

    std::vector<float> inputs;
    std::vector<float> head_pos;
    for(size_t i = 0; i < sizeof(trained_offsets) / sizeof(trained_offsets[0]); i++)
    {
        for(auto inputPath: input_device_paths)
        {
            auto mat = getDevicePose(inputPath);
            std::vector<float> flatmat{
                mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3] / 1.879f,
                mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3] / 1.879f,
                mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3] / 1.879f
            };
            // make input position head-relative
            if (i == 0) {
                head_pos = { flatmat[3], flatmat[7], flatmat[11] };
            }
            flatmat[3] -= head_pos[0];
            flatmat[7] -= head_pos[1];
            flatmat[11] -= head_pos[2];
            inputs.insert(inputs.end(), flatmat.begin(), flatmat.end());
        }
    }
    std::vector<float> transposed_inputs;
    // TODO: input data is transposed... should fix that at some point.
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            transposed_inputs.push_back(inputs[j * 12 + i]);
        }
    }

    // TODO: make one model.predict() update many trackers
    // format and feed into model.predict()
    // TODO: fail gracefully if we don't find fdeep_model.json
    // may actually be better to hardcode the JSON file into the driver
    const auto result = prediction_model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(12),
                                           static_cast<std::size_t>(9)),
        transposed_inputs)})[0];

    std::cout << fdeep::show_tensor(result) << std::endl;

    size_t y_out = (data_convention == input_convention::XYZ) ? 1 : 2;
    size_t z_out = (data_convention == input_convention::XYZ) ? 2 : 1;

    // convert result into vecPosition and qRotation
    // In the original dataset, most data gets boxed into a value of +-28 units, 
    // presumably inches? For now, hardcode the scale factor to my height (~6'2")
    // Also add head (first thing queried) position
    pose.vecPosition[0] = inputs[3] + result.get(fdeep::tensor_pos(device_offset, 3)) * 1.879;
    pose.vecPosition[y_out] = inputs[7] + result.get(fdeep::tensor_pos(device_offset, 7)) * 1.879;
    pose.vecPosition[z_out] = inputs[11] + result.get(fdeep::tensor_pos(device_offset, 11)) * 1.879;

    pose.vecPosition[2] -= 2.;

    GetDriver()->Log(std::to_string(pose.vecPosition[0]) + " " + std::to_string(pose.vecPosition[1]) + " " + std::to_string(pose.vecPosition[2]));

    Eigen::Matrix3f rotMatrix;
    rotMatrix(0,0) = result.get(fdeep::tensor_pos(device_offset, 0));
    rotMatrix(0,1) = result.get(fdeep::tensor_pos(device_offset, 1));
    rotMatrix(0,2) = result.get(fdeep::tensor_pos(device_offset, 2));

    // TODO: debug messages that complain if the X/Y/Z magnitude aren't roughly 1.

    // Swap Y and Z here
    rotMatrix(y_out,0) = result.get(fdeep::tensor_pos(device_offset, 4));
    rotMatrix(y_out,1) = result.get(fdeep::tensor_pos(device_offset, 5));
    rotMatrix(y_out,2) = result.get(fdeep::tensor_pos(device_offset, 6));

    rotMatrix(z_out,0) = result.get(fdeep::tensor_pos(device_offset, 8));
    rotMatrix(z_out,1) = result.get(fdeep::tensor_pos(device_offset, 9));
    rotMatrix(z_out,2) = result.get(fdeep::tensor_pos(device_offset, 10));

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
    pose.deviceIsConnected = true;
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
    if (device_offset == 0)
    {
        // waist
        GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String, "locator");
    }
    else if (device_offset > 2)
    {
        // elbows and knees
        GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String, "arrow");
    }
    else {
        // feet
        GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String, "vr_controller_05_wireless_b");
    }
    
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

/*int main()
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
}*/
