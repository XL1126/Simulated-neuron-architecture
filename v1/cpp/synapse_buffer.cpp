#include "synapse_buffer.h"

void SynapseSOA::reserve(size_t n) {
    weight.reserve(n);
    pre_neuron.reserve(n);
    post_neuron.reserve(n);
    tag.reserve(n);
    eligibility_trace.reserve(n);
    d1_density.reserve(n);
    d2_density.reserve(n);
    delay.reserve(n);
    is_core.reserve(n);
    ltp_e_fast.reserve(n);
    ltp_e_medium.reserve(n);
    ltp_e_slow.reserve(n);
    ltd_e_fast.reserve(n);
    ltd_e_medium.reserve(n);
    ltd_e_slow.reserve(n);
}

void SynapseSOA::clear() {
    weight.clear();
    pre_neuron.clear();
    post_neuron.clear();
    tag.clear();
    eligibility_trace.clear();
    d1_density.clear();
    d2_density.clear();
    delay.clear();
    is_core.clear();
    ltp_e_fast.clear();
    ltp_e_medium.clear();
    ltp_e_slow.clear();
    ltd_e_fast.clear();
    ltd_e_medium.clear();
    ltd_e_slow.clear();
}

void NeuronSOA::resize(size_t n) {
    v.resize(n, -65.0f);
    u.resize(n, -13.0f);
    I_syn.resize(n, 0.0f);
    I_dendrite.resize(n, 0.0f);
    I_bias.resize(n, 0.0f);
    I_noise.resize(n, 0.0f);
    threshold.resize(n, 30.0f);
    a.resize(n, 0.02f);
    b.resize(n, 0.2f);
    c.resize(n, -65.0f);
    d.resize(n, 8.0f);
    avg_firing_rate.resize(n, 0.0f);
    fired.resize(n, 0);
    r1.resize(n, 0.0f); r2.resize(n, 0.0f);
    o1.resize(n, 0.0f); o2.resize(n, 0.0f);
    ca_concentration.resize(n, 0.1f);
    out_begin.resize(n, -1);
    out_count.resize(n, 0);
    fire_count.resize(n, 0);
}