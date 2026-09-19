#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

struct PlannedAction {
    std::vector<float> action_state;
    std::vector<float> expected_outcome;
    float expected_free_energy;
    float expected_value;
    float confidence;
    int horizon;
};

struct BeliefState {
    std::vector<float> hidden_state;
    std::vector<float> precision;
    float free_energy;
    float model_evidence;
};

class ActiveInferenceEngine {
public:
    static constexpr int STATE_DIM = 32;
    static constexpr int ACTION_DIM = 16;
    static constexpr int MAX_PLANS = 8;
    static constexpr int MAX_HORIZON = 5;
    static constexpr int BELIEF_DIM = 32;

    PlannedAction action_plans[MAX_PLANS];
    int n_plans;
    int active_plan_idx;
    BeliefState current_belief;
    std::vector<float> prior_preferences;
    std::vector<float> expected_free_energy_history;
    float avg_free_energy;
    float planning_depth;
    float action_confidence;
    float exploration_bonus;
    uint64_t step_counter;

    ActiveInferenceEngine();

    void init(uint32_t seed);

    void update_beliefs(
        const std::vector<float>& observation,
        const std::vector<float>& action_taken,
        float prediction_error);

    void generate_plans(
        const std::vector<float>& current_state,
        const std::vector<float>& goal_state,
        const std::vector<float>& world_model_prediction,
        const std::vector<float>& emotion_vector,
        float dopamine,
        float curiosity);

    void evaluate_plans(
        const std::vector<float>& current_state,
        const std::vector<float>& goal_direction,
        float risk_aversion,
        float temporal_discount);

    void select_plan(float meta_confidence, float randomness);

    std::vector<float> get_planned_action() const;
    std::vector<float> get_predicted_outcome() const;
    std::string get_plan_description() const;

    float compute_expected_free_energy(
        const PlannedAction& plan,
        const std::vector<float>& goal) const;

    std::vector<float> get_policy_vector() const;

private:
    std::vector<float> _transition_model;
    void _update_transition_model(
        const std::vector<float>& prev_state,
        const std::vector<float>& action,
        const std::vector<float>& next_state,
        float lr);

    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};