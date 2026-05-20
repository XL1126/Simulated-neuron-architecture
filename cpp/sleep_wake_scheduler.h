#pragma once
#include <cstdint>
#include <cmath>

struct SleepWakeScheduler {
    enum SystemState {
        AWAKE_ONLINE,
        QUIET_REST,
        DEEP_SLEEP
    };

    SystemState current_state;
    float input_activity_level;
    int steps_since_last_sleep;

    static constexpr float REST_THRESHOLD = 0.15f;
    static constexpr float SLEEP_INTERVAL = 500.0f;

    SleepWakeScheduler();

    SystemState update_state(float sensory_input_magnitude,
                              bool force_sleep, bool force_awake);

    SystemState get_state() const { return current_state; }
    bool is_sleep_state() const { return current_state == DEEP_SLEEP || current_state == QUIET_REST; }
    bool is_awake() const { return current_state == AWAKE_ONLINE; }
    bool is_deep_sleep() const { return current_state == DEEP_SLEEP; }
    bool is_quiet_rest() const { return current_state == QUIET_REST; }

    float get_input_level() const { return input_activity_level; }
    int get_steps_since_sleep() const { return steps_since_last_sleep; }
};