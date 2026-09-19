#include "cortical_reactivation.h"
#include <algorithm>
#include <cstdlib>

static inline bool is_bad(float x) { return !std::isfinite(x); }

CorticalReactivationReceiver::CorticalReactivationReceiver()
    : ca1_size(0), cortex_size(0)
{}

void CorticalReactivationReceiver::init(size_t ca1_n, size_t cortical_n) {
    ca1_size = ca1_n;
    cortex_size = cortical_n;

    W_ca1_to_cortex.resize(ca1_n, std::vector<float>(cortical_n, 0.0f));
    for (size_t i = 0; i < ca1_n; i++) {
        for (size_t j = 0; j < cortical_n; j++) {
            uint32_t h = static_cast<uint32_t>(i * 123457 + j * 76543);
            W_ca1_to_cortex[i][j] = ((float)(h % 20000) / 10000.0f - 1.0f) * 0.05f;
        }
    }

    apical_calcium.resize(cortical_n, 0.1f);
}

void CorticalReactivationReceiver::receive_ca1_replay(
    const std::vector<float>& ca1_replay_output,
    bool is_swr_active, float swr_amplitude,
    std::vector<std::vector<float>>& cortical_weights,
    std::vector<float>& cortical_activity)
{
    if (ca1_replay_output.empty() || cortical_weights.empty()) return;
    if (is_bad(swr_amplitude)) swr_amplitude = 0.0f;

    for (size_t j = 0; j < cortex_size && j < apical_calcium.size(); j++) {
        float apical_input = 0.0f;
        for (size_t i = 0; i < ca1_size && i < ca1_replay_output.size(); i++) {
            if (i < W_ca1_to_cortex.size() && j < W_ca1_to_cortex[i].size()) {
                float w = W_ca1_to_cortex[i][j];
                float ca1 = ca1_replay_output[i];
                if (!is_bad(w) && !is_bad(ca1)) {
                    apical_input += w * ca1;
                }
            }
        }

        apical_calcium[j] += apical_input * swr_amplitude * 0.5f;
        apical_calcium[j] *= 0.95f;
        if (is_bad(apical_calcium[j])) apical_calcium[j] = 0.1f;

        if (apical_calcium[j] > APICAL_LTP_THRESHOLD) {
            for (size_t k = 0; k < cortex_size && k < cortical_activity.size(); k++) {
                if (cortical_activity[k] > 0.0f && k < cortical_weights.size()
                    && j < cortical_weights[k].size()) {
                    float delta = APICAL_LEARNING_RATE * cortical_activity[k]
                                  * cortical_activity[j];
                    if (!is_bad(delta)) {
                        cortical_weights[k][j] += delta;
                        float& w = cortical_weights[k][j];
                        if (is_bad(w)) w = 0.01f;
                        if (w > 0.5f) w = 0.5f;
                        if (w < -0.5f) w = -0.5f;
                    }
                }
            }
        }
    }
}

void CorticalReactivationReceiver::slow_cortical_consolidation(
    std::vector<std::vector<float>>& cortical_weights,
    std::vector<int>& coactivation_counts)
{
    for (size_t i = 0; i < cortical_weights.size(); i++) {
        for (size_t j = 0; j < cortical_weights[i].size(); j++) {
            if (i < coactivation_counts.size() && j < coactivation_counts.size()) {
                int count_i = coactivation_counts[i];
                int count_j = coactivation_counts[j];
                if (count_i > 10 && count_j > 10) {
                    float delta = BASAL_LEARNING_RATE * static_cast<float>(count_i + count_j);
                    if (!is_bad(delta)) {
                        cortical_weights[i][j] += delta;
                        float& w = cortical_weights[i][j];
                        if (w > 0.5f) w = 0.5f;
                        if (w < -0.5f) w = -0.5f;
                    }
                }
            }
        }
    }
}

void CorticalReactivationReceiver::update_apical_calcium_decay(float dt_ms) {
    for (auto& ca : apical_calcium) {
        ca = std::max(0.0f, ca - dt_ms * 0.01f);
        if (is_bad(ca)) ca = 0.1f;
    }
}