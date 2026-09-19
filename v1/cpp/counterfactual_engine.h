#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct CounterfactualOutcome {
    int action;
    float predicted_outcome;
    float regret;
};

struct CounterfactualEngine {
    static constexpr int N_ACTIONS = 8;
    static constexpr int HISTORY_LEN = 50;
    static constexpr int STATE_DIM = 64;

    std::deque<std::vector<float>> state_history;
    std::deque<int> action_history;
    std::deque<float> outcome_history;

    std::vector<float> action_value_model;
    std::vector<float> transition_weights;
    float regret_ema;
    float counterfactual_curiosity;
    int steps;

    CounterfactualEngine();

    void init();

    std::vector<CounterfactualOutcome> simulate_alternatives(
        const std::vector<float>& prev_state,
        int actual_action, float actual_outcome,
        const std::vector<float>& current_state,
        const std::vector<float>& world_prediction);

    void learn_from_regret(const std::vector<CounterfactualOutcome>& alternatives,
                            const std::vector<float>& state);

    float get_regret() const { return regret_ema; }
    float get_cf_curiosity() const { return counterfactual_curiosity; }

    void record_experience(const std::vector<float>& state,
                            int action, float outcome);
};