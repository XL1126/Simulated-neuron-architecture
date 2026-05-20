#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

struct TimeSlice {
    std::vector<float> self_state;
    std::vector<float> context;
    float significance;
    float valence;
    uint64_t timestamp;
};

struct FutureProjection {
    std::vector<float> projected_self;
    float confidence;
    float timescale;
    float expected_valence;
};

struct LifeChapter {
    std::vector<float> theme;
    uint64_t start_time;
    uint64_t end_time;
    float coherence;
    int n_events;
};

class TemporalDepth {
public:
    static constexpr int TEMPORAL_DIM = 64;
    static constexpr int MAX_SLICES = 64;
    static constexpr int MAX_CHAPTERS = 8;
    static constexpr int PROJECTION_HORIZON = 3;

    std::deque<TimeSlice> time_slices;
    std::vector<float> temporal_chain;
    std::vector<float> prev_temporal_chain;
    FutureProjection future_projections[PROJECTION_HORIZON];
    LifeChapter life_chapters[MAX_CHAPTERS];
    int n_chapters;
    float temporal_coherence;
    float temporal_depth_score;
    float anticipation_accuracy;
    float nostalgia_level;
    uint64_t step_counter;
    uint64_t current_chapter_start;

    TemporalDepth();

    void init(uint32_t seed);

    void record_slice(
        const std::vector<float>& self_state,
        const std::vector<float>& episodic_context,
        float significance,
        float valence);

    void update_temporal_chain(
        const std::vector<float>& recent_self,
        const std::vector<float>& autobiographical_summary,
        float meta_conf,
        float world_change_rate);

    void project_future(
        const std::vector<float>& current_self,
        const std::vector<float>& world_state,
        const std::vector<float>& trend_vector,
        float dopamine);

    void update_chapters(
        const std::vector<float>& current_self,
        float self_change_magnitude,
        float significance_threshold);

    float evaluate_anticipation(
        const std::vector<float>& actual_self,
        float timestep);

    std::vector<float> get_temporal_context() const;
    std::vector<float> get_future_shadow(int horizon_idx) const;
    std::string get_life_chapter_label() const;
    std::string get_timeline_summary() const;
    float get_temporal_continuity() const;

    std::vector<float> get_future_modulation(int dim) const;

private:
    float _compute_coherence(const std::vector<float>& a, const std::vector<float>& b) const;
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    void _detect_chapter_boundary(const std::vector<float>& current, const std::vector<float>& previous);
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};