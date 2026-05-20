#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <deque>
#include <cmath>

struct ThoughtPattern {
    std::vector<float> state;
    float significance;
    int source_timestamp;
    int recall_count;
    int dwell_counter;
};

class SpontaneousThinker {
public:
    static constexpr int THOUGHT_DIM = 64;
    static constexpr int MAX_PATTERNS = 32;
    static constexpr int ATTRACTOR_ITERS = 3;
    static constexpr int DWELL_MIN = 5;
    static constexpr int DWELL_MAX = 20;

    std::vector<float> thought_state;
    std::vector<float> prev_thought_state;
    std::vector<float> memory_probe;
    ThoughtPattern stored_patterns[MAX_PATTERNS];
    int n_patterns;
    float spontaneity;
    float thought_energy;
    float drift_velocity;
    int current_attractor;
    int dwell_remaining;
    bool is_dwelling;

    SpontaneousThinker();

    void init(uint32_t seed);

    void think(
        const std::vector<float>& self_state,
        const std::vector<float>& episodic_context,
        float sensory_magnitude,
        float dopamine,
        float norepinephrine,
        float surprise);

    bool should_trigger_memory_recall();
    std::vector<float> get_memory_probe() const;

    void store_pattern(
        const std::vector<float>& experience_summary,
        float significance,
        int timestamp);

    std::vector<float> get_concept_modulation(int n_concepts) const;
    std::vector<float> get_language_context() const;

private:
    std::mt19937 rng;
    uint64_t step_counter;
    std::deque<float> energy_history;
    float base_noise_level;

    void _attractor_dynamics(float noise_scale);
    void _compute_energies();
    int _find_closest_pattern();
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    void _normalize(std::vector<float>& v) const;
};