#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <deque>

struct SWREvent {
    int start_step;
    float duration_ms;
    float max_amplitude;
    float mean_frequency;
    int replay_length;
};

struct SWREngine {
    std::vector<float> pv_basket_activity;
    std::vector<float> anti_swr_activity;

    float swr_trigger_threshold;
    float swr_duration;
    float swr_amplitude;
    bool in_swr;
    int steps_since_last_swr;
    uint64_t swr_start_step;

    std::deque<SWREvent> recent_swrs;
    static constexpr size_t MAX_RECENT_SWRS = 50;

    static constexpr float PV_DECAY = 0.6f;
    static constexpr float ANTI_SWR_DECAY = 0.92f;
    static constexpr float SWR_RISE_TIME = 15.0f;
    static constexpr float SWR_FALL_TIME = 35.0f;
    static constexpr float SWR_TOTAL_DURATION = 50.0f;
    static constexpr float REPLAY_COMPRESSION = 15.0f;

    float threshold_multiplier;
    bool enabled;

    SWREngine();

    void init(size_t ca3_size);
    bool detect_swr_onset(const std::vector<float>& ca3_pyramidal_activity, uint64_t step);
    float update_swr(float dt_ms, uint64_t step);
    void set_threshold_multiplier(float m) { threshold_multiplier = m; }
    void enable(bool e) { enabled = e; }

    int get_replay_length() const;
    float get_ripple_oscillation(float elapsed_ms) const;

    struct SWRStats {
        int count;
        float mean_duration;
        float mean_amplitude;
        float replay_coverage;
    };
    SWRStats compute_stats() const;

private:
    float mean(const std::vector<float>& v) const;
};