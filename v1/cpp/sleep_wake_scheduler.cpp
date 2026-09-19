#include "sleep_wake_scheduler.h"
#include <algorithm>

static inline bool is_bad(float x) { return !std::isfinite(x); }

SleepWakeScheduler::SleepWakeScheduler()
    : current_state(AWAKE_ONLINE), input_activity_level(0.0f), steps_since_last_sleep(0)
{}

SleepWakeScheduler::SystemState SleepWakeScheduler::update_state(
    float sensory_input_magnitude, bool force_sleep, bool force_awake)
{
    if (is_bad(sensory_input_magnitude)) sensory_input_magnitude = 0.0f;

    input_activity_level = 0.95f * input_activity_level
                           + 0.05f * sensory_input_magnitude;

    steps_since_last_sleep++;

    if (force_awake) return (current_state = AWAKE_ONLINE);
    if (force_sleep) return (current_state = DEEP_SLEEP);

    if (static_cast<float>(steps_since_last_sleep) >= SLEEP_INTERVAL) {
        steps_since_last_sleep = 0;
        return (current_state = DEEP_SLEEP);
    }

    if (input_activity_level < REST_THRESHOLD) {
        return (current_state = QUIET_REST);
    }

    return (current_state = AWAKE_ONLINE);
}