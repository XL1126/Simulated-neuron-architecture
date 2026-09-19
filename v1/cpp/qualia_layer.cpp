#include "qualia_layer.h"
#include <random>
#include <chrono>

static inline bool is_bad(float x) { return !std::isfinite(x); }

QualiaLayer::QualiaLayer()
    : binding_strength(0.5f), perceptual_vividness(0.3f), first_person_salience(0.2f)
{}

void QualiaLayer::init() {
    H_state.assign(N_NEURONS, 0.0f);
    P_sensory.assign((size_t)N_NEURONS * (size_t)SENSORY_DIM, 0.0f);
    Q_readout.assign((size_t)N_NEURONS * (size_t)QUALIA_DIM, 0.0f);

    qualia_vector.assign(QUALIA_DIM, 0.0f);
    bound_percept.assign(SENSORY_DIM, 0.0f);
    raw_feel.assign(QUALIA_DIM, 0.0f);
    qualia_history.clear();

    binding_strength = 0.5f;
    perceptual_vividness = 0.3f;
    first_person_salience = 0.2f;

    std::mt19937 rng((unsigned)(std::chrono::steady_clock::now().time_since_epoch().count() + 777));
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    for (auto& v : P_sensory) v = dist(rng);
    for (auto& v : Q_readout) v = dist(rng);
    for (auto& v : H_state) v = dist(rng) * 0.1f;
    last_sensory_input.assign(SENSORY_DIM, 0.0f);
}

void QualiaLayer::forward(const std::vector<float>& visual,
                           const std::vector<float>& auditory,
                           const std::vector<float>& tactile,
                           const std::vector<float>& vestibular,
                           const std::vector<float>& self_state,
                           float global_dopamine, float global_norepinephrine,
                           float dt_ms) {

    (void)dt_ms;

    if (H_state.size() != (size_t)N_NEURONS) {
        H_state.assign(N_NEURONS, 0.0f);
    }
    if (qualia_vector.size() != (size_t)QUALIA_DIM) {
        qualia_vector.assign(QUALIA_DIM, 0.0f);
    }
    if (P_sensory.size() != (size_t)(N_NEURONS * SENSORY_DIM)) {
        P_sensory.assign(N_NEURONS * SENSORY_DIM, 0.0f);
    }
    if (Q_readout.size() != (size_t)(N_NEURONS * QUALIA_DIM)) {
        Q_readout.assign(N_NEURONS * QUALIA_DIM, 0.0f);
    }

    std::vector<float> sensory_input(SENSORY_DIM, 0.0f);
    size_t cursor = 0;

    auto safe_copy = [&](const float* src_ptr, size_t src_size, size_t max_n) {
        size_t room = (size_t)SENSORY_DIM - cursor;
        size_t n = src_size < max_n ? src_size : max_n;
        if (n > room) n = room;
        for (size_t i = 0; i < n; i++) {
            float v = src_ptr[i];
            if (is_bad(v)) v = 0.0f;
            v = std::max(-5.0f, std::min(5.0f, v));
            sensory_input[cursor + i] = std::tanh(v * 0.5f);
        }
        cursor += n;
    };

    float mod = 1.0f + std::max(0.0f, std::min(1.0f, global_dopamine)) * 0.15f
                + std::max(0.0f, std::min(1.0f, global_norepinephrine)) * 0.1f;

    safe_copy(visual.empty() ? nullptr : visual.data(), visual.size(), 20);
    safe_copy(auditory.empty() ? nullptr : auditory.data(), auditory.size(), 16);
    safe_copy(tactile.empty() ? nullptr : tactile.data(), tactile.size(), 14);
    safe_copy(vestibular.empty() ? nullptr : vestibular.data(), vestibular.size(), 10);
    safe_copy(self_state.empty() ? nullptr : self_state.data(), self_state.size(), 20);

    last_sensory_input = sensory_input;

    for (int ni = 0; ni < N_NEURONS; ni++) {
        float act = 0.0f;
        size_t base = (size_t)ni * (size_t)SENSORY_DIM;
        for (size_t si = 0; si < (size_t)SENSORY_DIM; si++) {
            size_t idx = base + si;
            float p = (idx < P_sensory.size()) ? P_sensory[idx] : 0.0f;
            act += sensory_input[si] * p;
        }
        float new_h = std::tanh(act * mod * 0.6f);
        if (!is_bad(new_h)) {
            H_state[ni] = H_state[ni] * 0.75f + new_h * 0.25f;
        }
        H_state[ni] = std::max(-1.0f, std::min(1.0f, H_state[ni]));
    }

    qualia_vector.assign(QUALIA_DIM, 0.0f);
    for (int qi = 0; qi < QUALIA_DIM; qi++) {
        float sum = 0.0f;
        for (int ni = 0; ni < N_NEURONS; ni++) {
            size_t idx = (size_t)ni * (size_t)QUALIA_DIM + (size_t)qi;
            float q = (idx < Q_readout.size()) ? Q_readout[idx] : 0.0f;
            sum += H_state[ni] * q;
        }
        qualia_vector[qi] = std::tanh(sum * 0.3f);
        if (is_bad(qualia_vector[qi])) qualia_vector[qi] = 0.0f;
    }

    bound_percept.assign(SENSORY_DIM, 0.0f);
    for (size_t si = 0; si < (size_t)SENSORY_DIM && si < sensory_input.size(); si++) {
        float feedback = 0.0f;
        for (int qi = 0; qi < QUALIA_DIM && qi < (int)qualia_vector.size(); qi++) {
            float q = qualia_vector[qi];
            if (is_bad(q)) q = 0.0f;
            float w = ((float)(((si * 7) + (qi * 13)) % 101) / 100.0f - 0.5f) * 0.04f;
            feedback += q * w;
        }
        float bp = sensory_input[si] * 0.6f + feedback;
        bound_percept[si] = std::max(-1.0f, std::min(1.0f, bp));
    }

    float activity = 0.0f;
    for (int ni = 0; ni < N_NEURONS; ni++) {
        if (!is_bad(H_state[ni])) activity += std::abs(H_state[ni]);
    }
    activity /= (float)N_NEURONS;
    perceptual_vividness = std::min(1.0f, activity * 3.0f + 0.08f);

    float local_coherence = 0.0f;
    for (int i = 0; i < N_NEURONS - 1; i++) {
        float diff = H_state[i] - H_state[i + 1];
        if (!is_bad(diff)) local_coherence += std::abs(diff);
    }
    local_coherence /= (float)(N_NEURONS) * 0.5f;
    binding_strength = std::max(0.0f, std::min(1.0f, 1.0f - local_coherence));

    float overlap = 0.0f;
    size_t n_overlap = self_state.size() < (size_t)QUALIA_DIM ? self_state.size() : (size_t)QUALIA_DIM;
    for (size_t i = 0; i < n_overlap; i++) {
        float ss = self_state[i];
        float qv = qualia_vector[i];
        if (is_bad(ss)) ss = 0.0f;
        if (is_bad(qv)) qv = 0.0f;
        overlap += std::abs(ss * qv);
    }
    if (n_overlap > 0) overlap /= (float)n_overlap;
    first_person_salience = std::min(1.0f, overlap * 1.8f + perceptual_vividness * 0.35f + 0.12f);

    qualia_history.push_back(qualia_vector);
    if ((int)qualia_history.size() > MAX_QUALIA_HISTORY) qualia_history.pop_front();

    raw_feel.assign(QUALIA_DIM, 0.0f);
    for (size_t i = 0; i < raw_feel.size() && i < qualia_vector.size(); i++) {
        raw_feel[i] = qualia_vector[i] * first_person_salience;
    }
}

void QualiaLayer::learn(float reward_signal) {
    float r = std::max(-1.0f, std::min(1.0f, reward_signal));
    float alpha = 0.001f * (1.0f + std::abs(r) * 0.5f);
    float mod = r > 0.0f ? 1.0f : 0.5f;

    for (int ni = 0; ni < N_NEURONS; ni++) {
        float h = H_state[ni];
        if (is_bad(h)) h = 0.0f;
        size_t base = (size_t)ni * (size_t)SENSORY_DIM;

        for (size_t si = 0; si < (size_t)SENSORY_DIM; si++) {
            size_t idx = base + si;
            if (idx >= P_sensory.size()) continue;
            float co_active = h * last_sensory_input[si];
            if (is_bad(co_active)) co_active = 0.0f;
            P_sensory[idx] += alpha * co_active * mod;
            P_sensory[idx] = std::max(-0.3f, std::min(0.3f, P_sensory[idx]));
        }
    }

    for (int qi = 0; qi < QUALIA_DIM; qi++) {
        float q = qualia_vector[qi];
        if (is_bad(q)) q = 0.0f;
        for (int ni = 0; ni < N_NEURONS; ni++) {
            float h = H_state[ni];
            if (is_bad(h)) h = 0.0f;
            size_t idx = (size_t)ni * (size_t)QUALIA_DIM + (size_t)qi;
            if (idx >= Q_readout.size()) continue;
            float co_active = h * q;
            if (is_bad(co_active)) co_active = 0.0f;
            Q_readout[idx] += alpha * co_active * mod;
            Q_readout[idx] = std::max(-0.3f, std::min(0.3f, Q_readout[idx]));
        }
    }
}