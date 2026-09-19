#include "stdp_engine.h"
#include <algorithm>

static inline bool is_bad(float x) { return !std::isfinite(x); }

float STDPEngine::compute_stdp_kernel(float dt_ms) const {
    float abs_dt = std::abs(dt_ms);
    if (abs_dt > 80.0f) return 0.0f;

    if (dt_ms > 0) {
        float ltp = A2_PLUS * std::exp(-abs_dt / 20.0f);
        return is_bad(ltp) ? 0.0f : ltp;
    }
    if (dt_ms < 0) {
        float ltd = -A3_MINUS * std::exp(-abs_dt / 20.0f);
        return is_bad(ltd) ? 0.0f : ltd;
    }
    return 0.0f;
}

float STDPEngine::compute_triplet_stdp(const NeuronState& pre,
                                         const NeuronState& post, float dt_ms) const {
    float r1 = is_bad(pre.r1) ? 0.0f : pre.r1;
    float r2 = is_bad(pre.r2) ? 0.0f : pre.r2;
    float o1 = is_bad(post.o1) ? 0.0f : post.o1;
    float o2 = is_bad(post.o2) ? 0.0f : post.o2;

    float ltd = A3_MINUS * o1 * (1.0f + r2);
    float ltp = A2_PLUS * r1 + A3_PLUS * r1 * o2;

    float result = ltp - ltd;
    return is_bad(result) ? 0.0f : std::max(-0.1f, std::min(0.1f, result));
}

void STDPEngine::decay_triplet_variables(NeuronState& neuron, float dt_ms) {
    float factor_r1 = std::exp(-dt_ms / TAU_R1);
    float factor_r2 = std::exp(-dt_ms / TAU_R2);
    float factor_o1 = std::exp(-dt_ms / TAU_O1);
    float factor_o2 = std::exp(-dt_ms / TAU_O2);

    neuron.r1 *= factor_r1; neuron.r2 *= factor_r2;
    neuron.o1 *= factor_o1; neuron.o2 *= factor_o2;

    if (is_bad(neuron.r1)) neuron.r1 = 0.0f;
    if (is_bad(neuron.r2)) neuron.r2 = 0.0f;
    if (is_bad(neuron.o1)) neuron.o1 = 0.0f;
    if (is_bad(neuron.o2)) neuron.o2 = 0.0f;
}

float STDPEngine::compute_dopamine_factor(const Synapse& syn, float da_signal) const {
    float d1 = syn.d1_density;
    float d2 = syn.d2_density;
    if (is_bad(d1)) d1 = 0.5f;
    if (is_bad(d2)) d2 = 0.5f;

    float d1_effect = d1 * std::max(0.0f, da_signal);
    float d2_effect = d2 * std::max(0.0f, -da_signal);
    return d1_effect + d2_effect;
}

float STDPEngine::compute_modulation_factor(float dopamine, float acetylcholine,
                                              float norepinephrine, float td_error) const {
    if (is_bad(dopamine)) dopamine = 0.0f;
    if (is_bad(acetylcholine)) acetylcholine = 0.0f;
    if (is_bad(norepinephrine)) norepinephrine = 0.0f;
    if (is_bad(td_error)) td_error = 0.0f;

    float factor = 1.0f + mu_d * dopamine + mu_a * acetylcholine
                   + 0.3f * norepinephrine + 0.8f * td_error;
    return std::max(0.0f, std::min(3.0f, factor));
}

void STDPEngine::apply_three_factor_update(Synapse& syn, float pre_post_event,
                                            float mod_signal, float lr) {
    if (is_bad(pre_post_event)) return;
    if (is_bad(mod_signal)) mod_signal = 1.0f;

    syn.ltp_cascade.update(std::max(0.0f, pre_post_event));
    syn.ltd_cascade.update(std::max(0.0f, -pre_post_event));

    float stdp_val = compute_stdp_kernel(pre_post_event * 20.0f);
    float eligibility = syn.eligibility_trace;
    if (is_bad(eligibility)) eligibility = 0.0f;

    float delta = stdp_val * eligibility * mod_signal * lr;
    if (is_bad(delta)) return;

    float& w = syn.weight;
    w += delta;
    if (is_bad(w)) w = 0.01f;
    if (w > 0.5f) w = 0.5f;
    if (w < -0.5f) w = -0.5f;
}

void STDPEngine::apply_triplet_three_factor(Synapse& syn,
                                              const NeuronState& pre, const NeuronState& post,
                                              float mod_signal, float dt_ms, float lr) {
    if (is_bad(mod_signal)) mod_signal = 1.0f;

    float triplet = compute_triplet_stdp(pre, post, dt_ms);
    if (is_bad(triplet)) return;

    syn.ltp_cascade.update(std::max(0.0f, triplet));
    syn.ltd_cascade.update(std::max(0.0f, -triplet));

    float credit = syn.ltp_cascade.get_credit();
    float mod = syn.pending_tag > 0.5f ? mod_signal * 2.0f : mod_signal * 0.5f;

    float delta = triplet * credit * mod * lr;
    if (is_bad(delta)) return;

    float& w = syn.weight;
    w += delta;
    if (is_bad(w)) w = 0.01f;
    if (w > 0.5f) w = 0.5f;
    if (w < -0.5f) w = -0.5f;
}

float STDPEngine::compute_calcium_modulated_update(float base_delta, float ca_factor) {
    if (is_bad(base_delta)) return 0.0f;
    if (is_bad(ca_factor)) ca_factor = 1.0f;
    return base_delta * ca_factor;
}

void STDPEngine::tag_synapse_for_evaluation(std::vector<Synapse>& synapses,
                                              const std::vector<uint32_t>& active_indices) {
    for (uint32_t idx : active_indices) {
        if (idx < synapses.size()) {
            synapses[idx].tag_for_evaluation();
        }
    }
}

void STDPEngine::resolve_pending_tags(std::vector<Synapse>& synapses,
                                        float reward_signal, float penalty_signal) {
    if (is_bad(reward_signal)) reward_signal = 0.0f;
    if (is_bad(penalty_signal)) penalty_signal = 0.0f;

    for (auto& syn : synapses) {
        if (syn.pending_tag < 0.1f) continue;

        if (reward_signal > 0.01f) {
            float ltp_credit = syn.ltp_cascade.get_credit();
            if (!is_bad(ltp_credit) && ltp_credit > 0.001f) {
                float delta = reward_signal * 0.005f * ltp_credit;
                syn.weight += delta;
                if (syn.weight > 0.5f) syn.weight = 0.5f;
            }
        }
        if (penalty_signal > 0.01f) {
            float ltd_credit = syn.ltd_cascade.get_credit();
            if (!is_bad(ltd_credit) && ltd_credit > 0.001f) {
                float delta = penalty_signal * 0.005f * ltd_credit;
                syn.weight -= delta;
                if (syn.weight < -0.5f) syn.weight = -0.5f;
            }
        }
        syn.pending_tag = 0.0f;
    }
}