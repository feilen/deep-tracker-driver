#pragma once

#include <chrono>
#include <cmath>
#include <openvr_capi.h>

#include <DriverFactory.hpp>
#include <DeviceType.hpp>
#include <IVRDevice.hpp>

namespace DeepTrackerDriver {
    class TrackerDevice : public IVRDevice {
        public:

            TrackerDevice(std::string serial);
            ~TrackerDevice() = default;

            // Inherited via IVRDevice
            virtual std::string GetSerial() override;
            virtual void Update() override;
            virtual vr::TrackedDeviceIndex_t GetDeviceIndex() override;
            virtual DeviceType GetDeviceType() override;

            virtual vr::EVRInitError Activate(uint32_t unObjectId) override;
            virtual void Deactivate() override;
            virtual void EnterStandby() override;
            virtual void* GetComponent(const char* pchComponentNameAndVersion) override;
            virtual void DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) override;
            virtual vr::DriverPose_t GetPose() override;

    private:
        vr::TrackedDeviceIndex_t device_index_ = vr::k_unTrackedDeviceIndexInvalid;
        std::string serial_;

        vr::DriverPose_t last_pose_ = IVRDevice::MakeDefaultPose();

        bool did_vibrate_ = false;
        float vibrate_anim_state_ = 0.f;

        vr::VRInputComponentHandle_t haptic_component_ = 0;

        vr::VRInputComponentHandle_t system_click_component_ = 0;
        vr::VRInputComponentHandle_t system_touch_component_ = 0;

        // These two must match the neural network inputs
        // TODO: make this user UserHandPrimary, configurable?
        const char * input_device_paths[3] = {
            k_pchPathUserHead,
            k_pchPathUserHandLeft,
            k_pchPathUserHandRight
        };

        // TODO: k_pchPathUserChest would be easy to add
        const char * output_device_paths[7] = {
            k_pchPathUserWaist,
            k_pchPathUserFootLeft,
            k_pchPathUserFootRight,
            k_pchPathUserElbowLeft,
            k_pchPathUserElbowRight,
            k_pchPathUserKneeLeft,
            k_pchPathUserKneeRight
        };
    };
};
