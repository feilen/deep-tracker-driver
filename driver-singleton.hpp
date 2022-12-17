#pragma once
#include <memory>

namespace DeepTrackerDriver {
    std::shared_ptr<DeepTrackerDriver::IVRDriver> GetDriver();
}
