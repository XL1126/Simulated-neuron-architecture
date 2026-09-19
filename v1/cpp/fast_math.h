#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace FastMath {

static constexpr int EXP_TABLE_SIZE = 10000;
static constexpr float EXP_TABLE_DT = 0.1f;
static constexpr float EXP_TABLE_MAX = (float)EXP_TABLE_SIZE * EXP_TABLE_DT;

inline void init_exp_table(std::vector<float>& table) {
    table.resize(EXP_TABLE_SIZE);
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        table[i] = std::exp(-(float)i * EXP_TABLE_DT);
    }
}

inline float fast_exp_decay(float dt_ms, float tau,
                              const std::vector<float>& exp_table) {
    float x = dt_ms / tau;
    if (x >= EXP_TABLE_MAX) return 0.0f;
    float idx_f = x / EXP_TABLE_DT;
    int idx = (int)idx_f;
    if (idx >= EXP_TABLE_SIZE - 1) return exp_table[EXP_TABLE_SIZE - 1];
    float frac = idx_f - (float)idx;
    return exp_table[idx] * (1.0f - frac) + exp_table[idx + 1] * frac;
}

inline float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

inline float fast_sigmoid(float x) {
    return 0.5f + 0.5f * fast_tanh(x * 0.5f);
}

inline void init_fast_math(std::vector<float>& exp_table) {
    init_exp_table(exp_table);
}

} // namespace FastMath