#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>

struct AbstractGoal {
    std::vector<float> goal_state;
    std::vector<float> goal_embedding;
    float value;
    float progress;
    float difficulty_estimate;
    float novelty;
    float competence_affordance;
    float self_consistency;
    float total_desirability;
    int timescale;
    uint64_t created_at;
    uint64_t last_pursued;
    int pursue_count;
    bool is_active;
    bool is_achieved;
};

class GoalGenerator {
public:
    static constexpr int GOAL_DIM = 32;
    static constexpr int EMBED_DIM = 32;
    static constexpr int MAX_GOALS = 16;
    static constexpr int SHORT_TERM = 0;
    static constexpr int MEDIUM_TERM = 1;
    static constexpr int LONG_TERM = 2;

    AbstractGoal active_goals[MAX_GOALS];
    int n_active_goals;
    int current_goal_idx;
    std::vector<float> goal_history_embedding;
    std::vector<float> achievement_memory;
    float goal_satisfaction;
    float goal_drive;
    float exploration_curiosity;
    float persistence;
    int consecutive_failures;
    uint64_t step_counter;
    std::deque<std::vector<float>> achieved_goal_embeddings;

    GoalGenerator();

    void init(uint32_t seed);

    void generate_candidates(
        const std::vector<float>& self_state,
        const std::vector<float>& world_state,
        const std::vector<float>& concept_activities,
        const std::vector<float>& temporal_context,
        float novelty_drive,
        float competence_drive,
        float self_consistency_drive,
        float curiosity);

    void evaluate_goals(
        const std::vector<float>& current_state,
        float reward_signal,
        float world_change);

    void select_active_goal(
        const std::vector<float>& self_state,
        float dopamine,
        float meta_conf);

    void update_progress(
        const std::vector<float>& current_state,
        const std::vector<float>& target_state);

    void mark_achieved(int goal_idx, float satisfaction);

    std::vector<float> get_active_goal_embedding() const;
    std::vector<float> get_goal_direction() const;
    std::string get_goal_description() const;
    float get_goal_progress() const;
    bool has_active_goal() const;
    std::string get_achievement_summary() const;
    std::vector<float> get_goal_context() const;

    void decay_stale_goals(uint64_t max_age);

private:
    std::mt19937 rng;
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    void _normalize(std::vector<float>& v) const;
    int _find_best_goal(const std::vector<float>& self_state) const;
    int _find_stalest_goal() const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};