#include "world_model.h"
#include <random>

static inline bool is_bad(float x) { return !std::isfinite(x); }

WorldModel::WorldModel()
    : world_prediction_error(0.5f)
{
    encoded.resize(HIDDEN_DIM, 0.0f);
    prediction.resize(SENSORY_DIM, 0.0f);
}

void WorldModel::init() {
    std::mt19937 rng(123);

    size_t enc_size = HIDDEN_DIM * SENSORY_DIM;
    W_enc.resize(enc_size);
    b_enc.resize(HIDDEN_DIM);
    std::normal_distribution<float> w_dist(0.0f, 0.05f);
    for (size_t i = 0; i < enc_size; i++) W_enc[i] = w_dist(rng);
    for (size_t i = 0; i < HIDDEN_DIM; i++) b_enc[i] = 0.0f;

    size_t pred_size = SENSORY_DIM * HIDDEN_DIM;
    W_pred.resize(pred_size);
    b_pred.resize(SENSORY_DIM);
    for (size_t i = 0; i < pred_size; i++) W_pred[i] = w_dist(rng);
    for (size_t i = 0; i < SENSORY_DIM; i++) b_pred[i] = 0.0f;
}

std::vector<float> WorldModel::predict_next(const std::vector<float>& current_sensory) {
    for (size_t o = 0; o < HIDDEN_DIM; o++) {
        float sum = b_enc[o];
        for (size_t i = 0; i < SENSORY_DIM && i < current_sensory.size(); i++) {
            float w = W_enc[o * SENSORY_DIM + i];
            float x = current_sensory[i];
            if (!is_bad(w) && !is_bad(x)) sum += w * x;
        }
        encoded[o] = std::tanh(sum);
    }

    for (size_t o = 0; o < SENSORY_DIM; o++) {
        float sum = b_pred[o];
        for (size_t i = 0; i < HIDDEN_DIM; i++) {
            float w = W_pred[o * HIDDEN_DIM + i];
            float h = encoded[i];
            if (!is_bad(w) && !is_bad(h)) sum += w * h;
        }
        prediction[o] = std::tanh(sum);
    }

    return prediction;
}

float WorldModel::compute_prediction_error(const std::vector<float>& predicted,
                                            const std::vector<float>& actual) const {
    float error = 0.0f;
    size_t n = std::min(predicted.size(), actual.size());
    for (size_t i = 0; i < n; i++) {
        float diff = predicted[i] - actual[i];
        if (!is_bad(diff)) error += diff * diff;
    }
    error /= std::max((float)n, 1.0f);
    return std::sqrt(std::max(0.0f, error));
}

void WorldModel::learn(const std::vector<float>& current_sensory,
                        const std::vector<float>& next_sensory, float lr) {
    auto pred = predict_next(current_sensory);
    world_prediction_error = compute_prediction_error(pred, next_sensory);

    for (size_t o = 0; o < SENSORY_DIM && o < next_sensory.size(); o++) {
        float error = next_sensory[o] - pred[o];
        if (is_bad(error)) continue;
        float delta = error * (1.0f - pred[o] * pred[o]);

        for (size_t i = 0; i < HIDDEN_DIM; i++) {
            float dw = delta * encoded[i] * lr;
            if (!is_bad(dw)) {
                W_pred[o * HIDDEN_DIM + i] += dw;
                W_pred[o * HIDDEN_DIM + i] = std::max(-1.0f, std::min(1.0f, W_pred[o * HIDDEN_DIM + i]));
            }
        }
    }
}

CLUBEstimator::CLUBEstimator()
    : current_mi(0.0f), buffer_pos(0)
{
    self_buffer.resize(BUFFER_SIZE);
    sensory_buffer.resize(BUFFER_SIZE);
}

void CLUBEstimator::init() {
    std::mt19937 rng(789);

    size_t w_self_size = CLUB_HIDDEN * SELF_DIM;
    W_self.resize(w_self_size);
    W_sensory.resize(CLUB_HIDDEN * SENSORY_ENC_DIM);
    b.resize(CLUB_HIDDEN);
    std::normal_distribution<float> w_dist(0.0f, 0.05f);
    for (size_t i = 0; i < w_self_size; i++) W_self[i] = w_dist(rng);
    for (size_t i = 0; i < CLUB_HIDDEN * SENSORY_ENC_DIM; i++) W_sensory[i] = w_dist(rng);
    for (size_t i = 0; i < CLUB_HIDDEN; i++) b[i] = 0.0f;
}

float CLUBEstimator::log_probability(const std::vector<float>& self,
                                       const std::vector<float>& sensory) const {
    float sum = b[0];
    for (size_t i = 0; i < CLUB_HIDDEN; i++) {
        float self_term = 0.0f;
        for (size_t j = 0; j < SELF_DIM && j < self.size(); j++) {
            float w = W_self[i * SELF_DIM + j];
            float s = self[j];
            if (!is_bad(w) && !is_bad(s)) self_term += w * s;
        }
        float sens_term = 0.0f;
        for (size_t j = 0; j < SENSORY_ENC_DIM && j < sensory.size(); j++) {
            float w = W_sensory[i * SENSORY_ENC_DIM + j];
            float v = sensory[j];
            if (!is_bad(w) && !is_bad(v)) sens_term += w * v;
        }
        sum += std::log(1.0f + std::exp(self_term + sens_term));
    }
    return is_bad(sum) ? 0.0f : sum;
}

float CLUBEstimator::estimate_mi(const std::vector<float>& self_state,
                                   const std::vector<float>& sensory_encoding) {
    float joint_lp = log_probability(self_state, sensory_encoding);

    self_buffer[buffer_pos] = self_state;
    sensory_buffer[buffer_pos] = sensory_encoding;
    buffer_pos = (buffer_pos + 1) % BUFFER_SIZE;

    current_mi = joint_lp;
    return current_mi;
}

float CLUBEstimator::compute_mi_upper_bound(const std::vector<float>& self_batch,
                                              const std::vector<float>& sensory_batch,
                                              size_t batch_size) {
    (void)self_batch;
    (void)sensory_batch;
    (void)batch_size;
    return std::max(0.0f, current_mi - 0.5f);
}

void CLUBEstimator::update_estimator(const std::vector<float>& self_state,
                                       const std::vector<float>& sensory_encoding, float lr) {
    float mi_val = estimate_mi(self_state, sensory_encoding);

    if (mi_val > 0.01f) {
        for (size_t i = 0; i < CLUB_HIDDEN; i++) {
            for (size_t j = 0; j < SELF_DIM && j < self_state.size(); j++) {
                float grad = mi_val * self_state[j] * lr * 0.1f;
                if (!is_bad(grad)) {
                    W_self[i * SELF_DIM + j] -= grad;
                    W_self[i * SELF_DIM + j] = std::max(-1.0f, std::min(1.0f, W_self[i * SELF_DIM + j]));
                }
            }
        }
    }
}