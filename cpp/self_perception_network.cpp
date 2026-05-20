#include "self_perception_network.h"
#include <random>

static inline bool is_bad(float x) { return !std::isfinite(x); }

SelfPerceptionNetwork::SelfPerceptionNetwork() {
    self_state.resize(SELF_DIM, 0.01f);
    h1.resize(HIDDEN_DIM, 0.0f);
    h1_norm.resize(HIDDEN_DIM, 0.0f);
    h1_act.resize(HIDDEN_DIM, 0.0f);
}

void SelfPerceptionNetwork::init() {
    std::mt19937 rng(42);

    size_t w1_size = HIDDEN_DIM * TOTAL_INPUT_DIM;
    W1.resize(w1_size);
    b1.resize(HIDDEN_DIM);
    std::normal_distribution<float> w_dist(0.0f, 0.02f);
    for (size_t i = 0; i < w1_size; i++) W1[i] = w_dist(rng);
    for (size_t i = 0; i < HIDDEN_DIM; i++) b1[i] = 0.0f;

    size_t w2_size = SELF_DIM * HIDDEN_DIM;
    W2.resize(w2_size);
    b2.resize(SELF_DIM);
    for (size_t i = 0; i < w2_size; i++) W2[i] = w_dist(rng);
    for (size_t i = 0; i < SELF_DIM; i++) b2[i] = 0.0f;

    ln1_gamma.resize(HIDDEN_DIM, 1.0f);
    ln1_beta.resize(HIDDEN_DIM, 0.0f);
}

void SelfPerceptionNetwork::fc_forward(const std::vector<float>& x, std::vector<float>& out,
                                         const std::vector<float>& W, const std::vector<float>& b,
                                         size_t in_dim, size_t out_dim) {
    for (size_t o = 0; o < out_dim; o++) {
        float sum = b[o];
        for (size_t i = 0; i < in_dim; i++) {
            float wi = W[o * in_dim + i];
            float xi = x[i];
            if (!is_bad(wi) && !is_bad(xi)) sum += wi * xi;
        }
        out[o] = sum;
    }
}

void SelfPerceptionNetwork::layer_norm(const std::vector<float>& x, std::vector<float>& out,
                                         const std::vector<float>& gamma,
                                         const std::vector<float>& beta) {
    size_t n = x.size();
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) { float v = x[i]; if (!is_bad(v)) mean += v; }
    mean /= (float)n;

    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - mean;
        if (!is_bad(diff)) var += diff * diff;
    }
    var /= (float)n;
    float inv_std = 1.0f / std::sqrt(std::max(var, 1e-5f));

    for (size_t i = 0; i < n; i++) {
        float norm = (x[i] - mean) * inv_std;
        out[i] = gamma[i] * norm + beta[i];
    }
}

void SelfPerceptionNetwork::relu(std::vector<float>& x) {
    for (auto& v : x) if (v < 0.0f) v = 0.0f;
}

std::vector<float> SelfPerceptionNetwork::forward(
    const std::vector<float>& motor_efference,
    const std::vector<float>& amygdala_emotion,
    const std::vector<float>& memory_recall,
    const std::vector<float>& prefrontal_intentions)
{
    std::vector<float> input(TOTAL_INPUT_DIM, 0.0f);

    size_t cursor = 0;
    auto copy_seg = [&](const std::vector<float>& src, size_t n) {
        for (size_t i = 0; i < n && i < src.size(); i++) {
            if (!is_bad(src[i])) input[cursor + i] = src[i];
        }
        cursor += n;
    };

    copy_seg(motor_efference, MOTOR_DIM);
    copy_seg(amygdala_emotion, AMYGDALA_DIM);
    copy_seg(memory_recall, MEMORY_DIM);
    copy_seg(prefrontal_intentions, INTENTION_DIM);

    fc_forward(input, h1, W1, b1, TOTAL_INPUT_DIM, HIDDEN_DIM);
    layer_norm(h1, h1_norm, ln1_gamma, ln1_beta);
    for (size_t i = 0; i < HIDDEN_DIM; i++) h1_act[i] = h1_norm[i];
    relu(h1_act);

    fc_forward(h1_act, self_state, W2, b2, HIDDEN_DIM, SELF_DIM);
    for (auto& v : self_state) v = std::tanh(v);

    return self_state;
}

void SelfPerceptionNetwork::train_step(const std::vector<float>& target, float lr) {
    if (target.size() != SELF_DIM) return;

    for (size_t i = 0; i < SELF_DIM; i++) {
        float error = target[i] - self_state[i];
        if (is_bad(error)) continue;

        float delta = error * (1.0f - self_state[i] * self_state[i]);

        for (size_t j = 0; j < HIDDEN_DIM; j++) {
            float dw = delta * h1_act[j] * lr;
            if (!is_bad(dw)) {
                W2[i * HIDDEN_DIM + j] += dw;
                W2[i * HIDDEN_DIM + j] = std::max(-1.0f, std::min(1.0f, W2[i * HIDDEN_DIM + j]));
            }
        }
    }
}