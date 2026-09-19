#pragma once
#include <vector>
#include <cstdint>
#include <cmath>

struct CorticalReactivationReceiver {
    std::vector<std::vector<float>> W_ca1_to_cortex;
    std::vector<float> apical_calcium;

    size_t ca1_size;
    size_t cortex_size;

    static constexpr float APICAL_LTP_THRESHOLD = 1.2f;
    static constexpr float BASAL_LEARNING_RATE = 0.0001f;
    static constexpr float APICAL_LEARNING_RATE = 0.002f;

    CorticalReactivationReceiver();
    void init(size_t ca1_n, size_t cortical_n);

    void receive_ca1_replay(const std::vector<float>& ca1_replay_output,
                             bool is_swr_active, float swr_amplitude,
                             std::vector<std::vector<float>>& cortical_weights,
                             std::vector<float>& cortical_activity);

    void slow_cortical_consolidation(std::vector<std::vector<float>>& cortical_weights,
                                      std::vector<int>& coactivation_counts);

    void update_apical_calcium_decay(float dt_ms);
};