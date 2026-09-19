#include "meta_cog.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

MetaCogLayer::MetaCogLayer()
    : conf(0.5f), surp(0.5f), valence(0.0f), arousal(0.3f),
      prev_summary_norm(0.0f), prev_conf(0.5f), homeostasis_target(0.1f)
{
    v.resize(N_NEURONS, -65.0f);
    u.resize(N_NEURONS, -13.0f);
    a.resize(N_NEURONS, 0.02f);
    b.resize(N_NEURONS, 0.2f);
    c.resize(N_NEURONS, -65.0f);
    d.resize(N_NEURONS, 8.0f);
    I_syn.resize(N_NEURONS, 0.0f);
    fired.resize(N_NEURONS, 0);
}

void MetaCogLayer::init() {
    std::mt19937 rng(777);
    std::normal_distribution<float> wd(0.0f, 0.05f);
    std::uniform_real_distribution<float> jitter(-2.0f, 2.0f);

    for (int i = 0; i < N_NEURONS; i++) {
        v[i] = -65.0f + jitter(rng);
        u[i] = b[i] * v[i];
        if (i < N_NEURONS / 4) {
            a[i] = 0.02f; b[i] = 0.2f; c[i] = -65.0f + jitter(rng); d[i] = 8.0f;
        } else if (i < N_NEURONS / 2) {
            a[i] = 0.02f; b[i] = 0.25f; c[i] = -55.0f + jitter(rng); d[i] = 0.05f;
        } else {
            a[i] = 0.02f; b[i] = 0.2f; c[i] = -50.0f + jitter(rng); d[i] = 2.0f;
        }
    }

    W_input.resize(N_NEURONS * SUMMARY_DIM);
    W_habits.resize(N_NEURONS);
    for (int i = 0; i < N_NEURONS; i++) {
        W_habits[i] = wd(rng) * 0.5f;
        for (int j = 0; j < SUMMARY_DIM; j++) {
            W_input[i * SUMMARY_DIM + j] = wd(rng);
        }
    }

    conf = 0.5f;
    surp = 0.5f;
    valence = 0.0f;
    arousal = 0.3f;
}

void MetaCogLayer::forward(const std::vector<float>& summary, float world_error, float self_error, float dt_ms) {
    (void)dt_ms;

    float summary_norm = 0.0f;
    for (auto x : summary) { if (!is_bad(x)) summary_norm += x * x; }
    summary_norm = std::sqrt(summary_norm + 1e-8f);

    float total_fire = 0.0f;
    for (int i = 0; i < N_NEURONS; i++) {
        fired[i] = 0;
        float syn = 0.0f;
        int base = i * SUMMARY_DIM;
        for (int j = 0; j < SUMMARY_DIM && j < (int)summary.size(); j++) {
            float w = W_input[base + j];
            float s = summary[j];
            if (!is_bad(w) && !is_bad(s)) syn += w * s;
        }
        float habit = W_habits[i];
        if (!is_bad(habit)) syn += habit * v[i] * 0.01f;
        if (is_bad(syn)) syn = 0.0f;

        I_syn[i] = syn;
        v[i] += 0.5f * (0.04f * v[i] * v[i] + 5.0f * v[i] + 140.0f - u[i] + I_syn[i]);
        v[i] += 0.5f * (0.04f * v[i] * v[i] + 5.0f * v[i] + 140.0f - u[i] + I_syn[i]);
        u[i] += a[i] * (b[i] * v[i] - u[i]);

        if (v[i] >= 30.0f) {
            v[i] = c[i];
            u[i] += d[i];
            fired[i] = 1;
            total_fire += 1.0f;
        }
        v[i] = std::max(-80.0f, std::min(40.0f, v[i]));
    }

    float fire_rate = total_fire / (float)N_NEURONS;

    float error_surprise = (world_error + self_error) * 3.0f;
    if (error_surprise > 1.0f) error_surprise = 1.0f;
    if (!std::isfinite(error_surprise)) error_surprise = 0.3f;

    float novelty_blend = error_surprise * 0.6f + fire_rate * 0.4f;

    surp = surp * 0.7f + novelty_blend * 0.3f;
    conf = conf * 0.7f + (1.0f - surp) * 0.3f;

    if (is_bad(conf)) conf = 0.5f;
    if (is_bad(surp)) surp = 0.5f;

    conf = std::max(0.0f, std::min(1.0f, conf));
    surp = std::max(0.0f, std::min(1.0f, surp));

    valence = valence * 0.95f + (surp < 0.3f ? 0.1f : surp > 0.6f ? -0.1f : 0.0f);
    valence = std::max(-1.0f, std::min(1.0f, valence));
    arousal = arousal * 0.9f + surp * 0.2f;
    arousal = std::max(0.0f, std::min(1.0f, arousal));

    prev_summary_norm = summary_norm;
    prev_conf = conf;
}

void MetaCogLayer::learn(const std::vector<float>& actual_summary, float reward_signal) {
    for (int i = 0; i < N_NEURONS; i++) {
        if (!fired[i]) continue;
        float lr = 0.001f * (1.0f + reward_signal * 0.5f);
        W_habits[i] += lr * (homeostasis_target - 0.05f);
        W_habits[i] = std::max(-0.5f, std::min(0.5f, W_habits[i]));
    }
    (void)actual_summary;
}