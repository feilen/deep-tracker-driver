#include <logging.hpp>

static std::function<void(std::string)> logging_fn;

void DeepTrackerDriver::SetLoggingFunction(std::function<void(std::string)> fn) {
	logging_fn = fn;
}

void DeepTrackerDriver::Log(std::string log) {
	if (logging_fn) {
		logging_fn(log);
	}
}
