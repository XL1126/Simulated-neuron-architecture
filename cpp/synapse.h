#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>

struct EligibilityCascade {
    float e_fast;
    float e_medium;
    float e_slow;

    static constexpr float GAMMA_FAST = 0.95f;
    static constexpr float GAMMA_MEDIUM = 0.99f;
    static constexpr float GAMMA_SLOW = 0.999f;

    EligibilityCascade() : e_fast(0.0f), e_medium(0.0f), e_slow(0.0f) {}

    void update(float spike_event) {
        e_fast   = GAMMA_FAST   * e_fast   + spike_event;
        e_medium = GAMMA_MEDIUM * e_medium + e_fast;
        e_slow   = GAMMA_SLOW   * e_slow   + e_medium;
    }

    float get_credit() const { return e_slow; }
    float get_medium() const { return e_medium; }
    float get_fast() const { return e_fast; }

    void decay_all(float factor) {
        e_fast *= factor; e_medium *= factor; e_slow *= factor;
    }
};

struct Synapse {
    uint32_t target_id;
    float weight;
    uint8_t delay;
    uint64_t last_use_step;
    bool is_core;
    float stdp_trace;

    float u;
    float x;
    float tau_facil;
    float tau_rec;
    float U_base;

    float eligibility_trace;

    float d1_density;
    float d2_density;

    EligibilityCascade ltp_cascade;
    EligibilityCascade ltd_cascade;

    float pending_tag;

    Synapse() : target_id(0), weight(0.0f), delay(1), last_use_step(0),
                is_core(false), stdp_trace(0.0f), u(0.0f), x(1.0f),
                tau_facil(100.0f), tau_rec(800.0f), U_base(0.5f),
                eligibility_trace(0.0f),
                d1_density(0.7f), d2_density(0.3f),
                pending_tag(0.0f) {}

    float compute_stp_effect() const { return u * x; }

    void update_stp_on_spike() {
        u = u + U_base * (1.0f - u);
        x = x - u * x;
        if (x < 0.0f) x = 0.0f;
        if (u > 1.0f) u = 1.0f;
    }

    void decay_stp(float dt_ms) {
        float du = -(u / tau_facil) * (dt_ms / 1.0f);
        float dx = ((1.0f - x) / tau_rec) * (dt_ms / 1.0f);
        u += du; x += dx;
        if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
        if (x < 0.0f) x = 0.0f; if (x > 1.0f) x = 1.0f;
    }

    void tag_for_evaluation() { pending_tag = 1.0f; }
    void decay_tag(float dt_ms) {
        if (pending_tag > 0.0f) pending_tag = std::max(0.0f, pending_tag - dt_ms * 0.01f);
    }
};