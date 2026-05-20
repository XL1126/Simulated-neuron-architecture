#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <deque>

struct NarrativeSelf {
    static constexpr int NARRATIVE_DIM = 128;
    static constexpr int HISTORY_DIM = 8;
    static constexpr int MAX_HISTORY = 256;

    std::vector<float> narrative;
    std::vector<float> baseline_self;
    std::deque<std::vector<float>> episodic_history;
    std::deque<float> valence_history;
    std::deque<float> conf_history;

    float narrative_inertia;
    float identity_stability;
    float recent_surprise_ema;
    int steps_alive;

    std::deque<std::string> causal_log;
    float prev_stability;
    std::string prev_label;

    NarrativeSelf();

    void init();

    void update(const std::vector<float>& self_state,
                const std::vector<float>& episodic_summary,
                float meta_conf, float meta_surp,
                float meta_valence, float intrinsic_reward);

    const std::vector<float>& who_am_i() const { return narrative; }
    float get_stability() const { return identity_stability; }
    float get_recent_surprise() const { return recent_surprise_ema; }
    std::string describe_identity() const;
    std::string get_causal_narrative() const;
};