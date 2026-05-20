#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct ActionSelector {
    static constexpr size_t N_ACTIONS = 8;
    static constexpr size_t SENSORY_DIM = 64;
    static constexpr int HISTORY_LEN = 20;

    std::vector<float> action_effects;
    std::vector<int> action_counts;
    std::vector<float> action_rewards;
    std::vector<std::vector<float>> recent_sensory;
    std::vector<int> recent_actions;
    int history_pos;
    int history_fill;

    float epsilon;
    float learning_rate;
    int steps;

    std::mt19937 rng;

    ActionSelector();

    void init();

    int select_action(const std::vector<float>& current_sensory,
                       const std::vector<float>& world_prediction,
                       float curiosity, float dopamine);

    void learn(const std::vector<float>& prev_sensory,
                const std::vector<float>& actual_next,
                int action_taken, float reward);

    float compute_efe(const std::vector<float>& predicted_next,
                       const std::vector<float>& target,
                       int action, float curiosity) const;

    std::vector<float> predict_action_outcome(
        const std::vector<float>& world_prediction, int action) const;
};