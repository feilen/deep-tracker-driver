#pragma once
#include <functional>
#include <string>
#include <memory>

namespace DeepTrackerDriver {
	void SetLoggingFunction(std::function<void(std::string)> fn);
	void Log(std::string log);
}