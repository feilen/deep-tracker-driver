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

    // input is:
    // (devices * frames) devices of
    // 3 rows of
    // 4 columns
    Eigen::Matrix4f head_correction_mat;
    auto input_tensor = fdeep::tensor(fdeep::tensor_shape(
        sizeof(input_device_paths) / sizeof(input_device_paths[0]) *
        sizeof(trained_offsets) / sizeof(trained_offsets[0]),
        static_cast<std::size_t>(3),
        static_cast<std::size_t>(4)),
        0.f);
    for(size_t i = 0; i < sizeof(trained_offsets) / sizeof(trained_offsets[0]); i++)
    {
        for(size_t j = 0; j < sizeof(input_device_paths) / sizeof(input_device_paths[0]); j++)
        {
            auto mat = getDevicePose(input_device_paths[j]);
            if (i == 0) {
                // Create a rotation matrix that makes everything relative to the head, locked to the Y axis
                Eigen::Matrix3f head_rotation_matrix {
                    {mat.m[0][0], mat.m[0][1], mat.m[0][2]},
                    {mat.m[1][0], mat.m[1][1], mat.m[1][2]},
                    {mat.m[2][0], mat.m[2][1], mat.m[2][2]}
                };
                Eigen::Vector3f eulerAngles = head_rotation_matrix.eulerAngles(0, 1, 2);
                head_correction_mat.block<3,3>(0,0) = Eigen::AngleAxisf(-eulerAngles[1], Eigen::Vector3f::UnitY()).toRotationMatrix();
                head_correction_mat(0, 3) = -mat.m[0][3] / SCALE_FACTOR;
                head_correction_mat(2, 3) = -mat.m[2][3] / SCALE_FACTOR;
                head_correction_mat(3, 3) = 1.f;
            }
            Eigen::Matrix4f device_matrix{
                {mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3] / SCALE_FACTOR},
                {mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3] / SCALE_FACTOR},
                {mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3] / SCALE_FACTOR},
                {        0.f,         0.f,         0.f,         1.f}
            };
            device_matrix *= head_correction_mat;

            for (size_t k = 0; k < 3; k++)
            {
                for (size_t l = 0; l < 4; l++)
                {
                    input_tensor.set(fdeep::tensor_pos(i * 3 + j, k, l), mat.m[k][l]);
                }
            }
        }
    }

    // TODO: make one model.predict() update many trackers
    // format and feed into model.predict()
    // TODO: hardcode the JSON file into the driver
    try {
        const auto result = prediction_model.predict({input_tensor})[0];

        GetDriver()->Log(fdeep::show_tensor(result));

        Eigen::Matrix4f result_matrix{
            {result.get(fdeep::tensor_pos(device_offset, 0, 0)), result.get(fdeep::tensor_pos(device_offset, 0, 1)), result.get(fdeep::tensor_pos(device_offset, 0, 2)), result.get(fdeep::tensor_pos(device_offset, 0, 3))},
            {result.get(fdeep::tensor_pos(device_offset, 1, 0)), result.get(fdeep::tensor_pos(device_offset, 1, 1)), result.get(fdeep::tensor_pos(device_offset, 1, 2)), result.get(fdeep::tensor_pos(device_offset, 1, 3))},
            {result.get(fdeep::tensor_pos(device_offset, 2, 0)), result.get(fdeep::tensor_pos(device_offset, 2, 1)), result.get(fdeep::tensor_pos(device_offset, 2, 2)), result.get(fdeep::tensor_pos(device_offset, 2, 3))},
            {                                               0.f,                                                0.f,                                                0.f,                                                1.f}
        };
        result_matrix *= head_correction_mat.inverse();

        // convert result into vecPosition and qRotation
        // In the original dataset, most data gets boxed into a value of +-28 units,
        // presumably inches? For now, hardcode the scale factor to my height (~6'2")
        // Also add head (first thing queried) position
        pose.vecPosition[0] = result_matrix(0, 3) * SCALE_FACTOR;
        pose.vecPosition[1] = result_matrix(1, 3) * SCALE_FACTOR;
        pose.vecPosition[2] = result_matrix(2, 3) * SCALE_FACTOR;

        GetDriver()->Log(std::to_string(pose.vecPosition[0]) + " " +
            std::to_string(pose.vecPosition[1]) + " " +
            std::to_string(pose.vecPosition[2]));

        Eigen::Quaternion<float> poseQuat(result_matrix.block<3,3>(0,0));
        poseQuat.normalize();
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
        pose.poseIsValid = true;

        // Post pose
        GetDriver()->GetDriverHost()->TrackedDevicePoseUpdated(this->device_index_,
            pose, sizeof(vr::DriverPose_t));
        this->last_pose_ = pose;
    }
    catch (std::runtime_error& e) {
        GetDriver()->Log(e.what());
        return;
    }
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

    //GetDriver()->GetInput()->CreateBooleanComponent(props, "/input/system/click", &this->system_click_component_);
    //GetDriver()->GetInput()->CreateBooleanComponent(props, "/input/system/touch", &this->system_touch_component_);

    // Set some universe ID (Must be 2 or higher)
    GetDriver()->GetProperties()->SetUint64Property(props, vr::Prop_CurrentUniverseId_Uint64, 2);

    // Set up a model "number" (not needed but good to have)
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_ModelNumber_String, "deep-tracker");

    // Opt out of hand selection
    GetDriver()->GetProperties()->SetInt32Property(props, vr::Prop_ControllerRoleHint_Int32, vr::ETrackedControllerRole::TrackedControllerRole_OptOut);

    // Set up a render model path
    /*if (device_offset == 0)
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
    }*/
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String, "cube_indicator");

    // Set controller profile
    GetDriver()->GetProperties()->SetStringProperty(props, vr::Prop_InputProfilePath_String, "{example}/input/deep_tracker_input.json");

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
