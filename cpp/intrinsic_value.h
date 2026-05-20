#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <deque>

struct IntrinsicValueNucleus {
    static constexpr int HISTORY_LEN = 30;
    std::deque<float> error_history;
    std::deque<float> reward_history;
    std::vector<float> self_reference;
    float prev_error;
    int steps;

    IntrinsicValueNucleus();

    void init(size_t self_dim);

    float compute(float prediction_error, float current_self_error,
                   const std::vector<float>& self_state);
};