#pragma once
#include "synapse.h"
#include "neuron_izhikevich.h"
#include <vector>
#include <cmath>

struct STDPConfig {
    float a_plus = 0.008f;
    float a_minus = 0.006f;
    float tau_plus_ms = 20.0f;
    float tau_minus_ms = 20.0f;
    float dopamine_k = 1.0f;
    float history_window_ms = 80.0f;
};

inline void stdp_update_synapse(Synapse& syn, uint64_t t_pre, uint64_t t_post,
                                  float dopamine, const STDPConfig& cfg) {
    float dt = (float)((int64_t)t_post - (int64_t)t_pre);
    float abs_dt = std::abs(dt);
    if (abs_dt > cfg.history_window_ms || abs_dt < 0.5f) return;

    float stdp_val;
    if (t_post > t_pre) {
        stdp_val = cfg.a_plus * std::exp(-abs_dt / cfg.tau_plus_ms);
    } else {
        stdp_val = -cfg.a_minus * std::exp(-abs_dt / cfg.tau_minus_ms);
    }

    float da_factor = 1.0f + cfg.dopamine_k * (dopamine - 0.5f);
    da_factor = std::max(0.0f, std::min(2.0f, da_factor));

    float delta = stdp_val * da_factor * 0.1f;
    syn.weight += delta;
    if (!std::isfinite(syn.weight)) syn.weight = 0.01f;
    if (syn.weight > 0.5f) syn.weight = 0.5f;
    if (syn.weight < -0.5f) syn.weight = -0.5f;
}

inline void stdp_update_eligibility(Synapse& syn, uint64_t t_pre, uint64_t t_post, float lambda) {
    float dt = std::abs((float)((int64_t)t_post - (int64_t)t_pre));
    float trace_update = std::exp(-dt / 30.0f);
    syn.eligibility_trace = lambda * syn.eligibility_trace + trace_update * 0.1f;
    if (!std::isfinite(syn.eligibility_trace)) syn.eligibility_trace = 0.0f;
}

inline void stdp_apply_credit(Synapse& syn, float reward, float eta) {
    float delta = reward * syn.eligibility_trace * eta;
    syn.weight += delta;
    if (!std::isfinite(syn.weight)) syn.weight = 0.01f;
    if (syn.weight > 0.5f) syn.weight = 0.5f;
    if (syn.weight < -0.5f) syn.weight = -0.5f;
    syn.eligibility_trace *= 0.95f;
}

class STDPEngine {
public:
    static constexpr float DEFAULT_LR = 0.001f;
    static constexpr float A2_PLUS = 0.008f;
    static constexpr float A3_PLUS = 0.006f;
    static constexpr float A3_MINUS = 0.007f;
    static constexpr float TAU_R1 = 15.0f;
    static constexpr float TAU_R2 = 50.0f;
    static constexpr float TAU_O1 = 25.0f;
    static constexpr float TAU_O2 = 80.0f;

    STDPEngine() = default;

    float compute_stdp_kernel(float dt_ms) const;
    float compute_triplet_stdp(const NeuronState& pre, const NeuronState& post, float dt_ms) const;

    void decay_triplet_variables(NeuronState& neuron, float dt_ms);

    float compute_dopamine_factor(const Synapse& syn, float da_signal) const;

    float compute_modulation_factor(float dopamine, float acetylcholine,
                                     float norepinephrine, float td_error) const;

    void apply_three_factor_update(Synapse& syn, float pre_post_event,
                                    float mod_signal, float lr = DEFAULT_LR);

    void apply_triplet_three_factor(Synapse& syn,
                                     const NeuronState& pre, const NeuronState& post,
                                     float mod_signal, float dt_ms, float lr = DEFAULT_LR);

    float compute_calcium_modulated_update(float base_delta, float ca_factor);

    void tag_synapse_for_evaluation(std::vector<Synapse>& synapses,
                                     const std::vector<uint32_t>& active_indices);
    void resolve_pending_tags(std::vector<Synapse>& synapses, float reward_signal,
                               float penalty_signal);

    float mu_d; float mu_a;
    float lr_global;

private:
    uint64_t last_step_count; uint64_t update_count;
};