#include "swr_engine.h"
#include <algorithm>
#include <numeric>

static inline bool is_bad(float x) { return !std::isfinite(x); }

SWREngine::SWREngine()
    : swr_trigger_threshold(0.15f), swr_duration(0.0f), swr_amplitude(0.0f),
      in_swr(false), steps_since_last_swr(0), swr_start_step(0),
      threshold_multiplier(1.0f), enabled(true)
{}

void SWREngine::init(size_t ca3_size) {
    pv_basket_activity.resize(ca3_size, 0.0f);
    anti_swr_activity.resize(ca3_size, 0.5f);
    steps_since_last_swr = static_cast<int>(ca3_size * 2);
    threshold_multiplier = 1.0f;
    enabled = true;
}

float SWREngine::mean(const std::vector<float>& v) const {
    if (v.empty()) return 0.0f;
    float s = 0.0f;
    for (float x : v) { if (!is_bad(x)) s += x; }
    return s / static_cast<float>(v.size());
}

bool SWREngine::detect_swr_onset(const std::vector<float>& ca3_pyramidal_activity,
                                   uint64_t step) {
    if (!enabled) return false;
    if (in_swr) return false;

    float avg_anti_swr = mean(anti_swr_activity);
    if (avg_anti_swr > 0.3f) return false;

    float avg_ca3 = mean(ca3_pyramidal_activity);
    float eff_threshold = swr_trigger_threshold * threshold_multiplier;
    if (avg_ca3 < eff_threshold) return false;

    if (steps_since_last_swr < 500) return false;

    if (ca3_pyramidal_activity.size() != pv_basket_activity.size()) return false;

    in_swr = true;
    swr_duration = SWR_TOTAL_DURATION;
    swr_amplitude = 0.0f;
    steps_since_last_swr = 0;
    swr_start_step = step;

    return true;
}

float SWREngine::update_swr(float dt_ms, uint64_t step) {
    (void)step;
    if (!in_swr) {
        steps_since_last_swr++;
        for (auto& a : anti_swr_activity) a = ANTI_SWR_DECAY * a + (1.0f - ANTI_SWR_DECAY) * 0.1f;
        for (auto& p : pv_basket_activity) p *= PV_DECAY;
        return 0.0f;
    }

    swr_duration -= dt_ms;

    float elapsed = SWR_TOTAL_DURATION - swr_duration;
    if (elapsed < SWR_RISE_TIME) {
        swr_amplitude = elapsed / SWR_RISE_TIME;
    } else if (swr_duration < SWR_FALL_TIME) {
        swr_amplitude = swr_duration / SWR_FALL_TIME;
    } else {
        swr_amplitude = 1.0f;
    }

    float ripple = swr_amplitude * std::sin(2.0f * 3.14159265f * 200.0f * elapsed / 1000.0f);

    if (swr_duration <= 0.0f) {
        SWREvent evt;
        evt.start_step = static_cast<int>(swr_start_step);
        evt.duration_ms = SWR_TOTAL_DURATION;
        evt.max_amplitude = 1.0f;
        evt.mean_frequency = 200.0f;
        evt.replay_length = get_replay_length();
        recent_swrs.push_back(evt);
        while (recent_swrs.size() > MAX_RECENT_SWRS) recent_swrs.pop_front();

        in_swr = false;
        swr_amplitude = 0.0f;
        steps_since_last_swr = 0;
    }

    return ripple;
}

int SWREngine::get_replay_length() const {
    if (recent_swrs.empty()) return 0;
    return recent_swrs.back().replay_length;
}

float SWREngine::get_ripple_oscillation(float elapsed_ms) const {
    return std::sin(2.0f * 3.14159265f * 200.0f * elapsed_ms / 1000.0f);
}

SWREngine::SWRStats SWREngine::compute_stats() const {
    SWRStats stats;
    stats.count = static_cast<int>(recent_swrs.size());
    stats.mean_duration = 0.0f;
    stats.mean_amplitude = 0.0f;
    stats.replay_coverage = 0.0f;

    if (recent_swrs.empty()) return stats;

    for (auto& e : recent_swrs) {
        stats.mean_duration += e.duration_ms;
        stats.mean_amplitude += e.max_amplitude;
        stats.replay_coverage += static_cast<float>(e.replay_length);
    }
    stats.mean_duration /= static_cast<float>(recent_swrs.size());
    stats.mean_amplitude /= static_cast<float>(recent_swrs.size());
    stats.replay_coverage /= static_cast<float>(recent_swrs.size());

    return stats;
}