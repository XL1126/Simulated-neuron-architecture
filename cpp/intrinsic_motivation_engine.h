#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct CuriosityOption {
    int target_kind;
    int target_index;
    float novelty_score;
    float info_gain;
    float competence_gain;
    float total_score;
};

class IntrinsicMotivationEngine {
public:
    static constexpr int N_REGIONS = 10;
    static constexpr int PREDICTOR_DIM = 8;
    static constexpr int HISTORY_LEN = 30;
    static constexpr int N_OPTIONS = 8;
    static constexpr int AMA_SAMPLE_N = 32;

    std::vector<float> region_ema[N_REGIONS];
    std::vector<float> region_predictor[N_REGIONS];
    std::vector<float> region_pred_error;
    std::vector<float> region_novelty;

    std::deque<float> global_error_history;
    std::deque<float> reward_history;

    float novelty_drive;
    float competence_drive;
    float self_consistency_drive;
    float composite_intrinsic_reward;
    float td_error;
    float td_error_ema;
    float prev_value_estimate;

    std::vector<float> self_reference;
    int steps;

    CuriosityOption candidate_options[N_OPTIONS];
    float exploration_temperature;
    float exploitation_ratio;

    IntrinsicMotivationEngine();

    void init(size_t self_dim, uint32_t seed);

    void update_region_prediction(int region_idx,
        const std::vector<float>& region_activity, size_t n_dims);

    float compute( float world_pred_error, float self_model_error,
        const std::vector<float>& self_state,
        const std::vector<float>& ama_self_prototype,
        float external_reward);

    void generate_curiosity_options(
        const std::vector<std::vector<float>>& region_states,
        const std::vector<int>& memory_indices);

    int select_curiosity_action(float temperature);

    float compute_novelty_bonus(const std::vector<float>& state,
        const std::vector<float>& history_ema);

    float compute_info_gain(const std::vector<float>& state,
        const std::vector<float>& predictor);

    float get_td_error() const { return td_error; }
    float get_novelty_drive() const { return novelty_drive; }
    float get_intrinsic_reward() const { return composite_intrinsic_reward; }
};