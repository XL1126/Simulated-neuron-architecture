#include "intrinsic_value.h"

IntrinsicValueNucleus::IntrinsicValueNucleus()
    : prev_error(0.5f), steps(0)
{}

void IntrinsicValueNucleus::init(size_t self_dim) {
    error_history.clear();
    reward_history.clear();
    self_reference.assign(self_dim, 0.0f);
    prev_error = 0.5f;
    steps = 0;
}

float IntrinsicValueNucleus::compute(float prediction_error, float current_self_error,
                                      const std::vector<float>& self_state) {
    steps++;

    error_history.push_back(prediction_error);
    if ((int)error_history.size() > HISTORY_LEN) error_history.pop_front();

    float novelty = 0.0f;
    if (error_history.size() >= 5) {
        float mean_err = 0.0f;
        for (auto e : error_history) mean_err += e;
        mean_err /= (float)error_history.size();
        novelty = std::abs(prediction_error - mean_err) / std::max(0.01f, mean_err);
        novelty = std::min(1.0f, novelty);
    }

    float progress = 0.0f;
    if (error_history.size() >= 10) {
        auto it = error_history.end();
        float recent = 0.0f;
        for (int i = 0; i < 5 && i < (int)error_history.size(); i++) {
            recent += *(--it);
        }
        recent /= 5.0f;
        float older = 0.0f;
        for (int i = 5; i < 10 && i < (int)error_history.size(); i++) {
            older += *(--it);
        }
        older /= 5.0f;
        progress = (older - recent) / std::max(0.01f, older + recent);
        if (progress < 0.0f) progress = 0.0f;
        if (progress > 1.0f) progress = 1.0f;
    }

    float consistency = 0.5f;
    if (self_reference.size() == self_state.size() && steps > 10) {
        float dist = 0.0f;
        for (size_t i = 0; i < self_state.size(); i++) {
            float diff = self_state[i] - self_reference[i];
            dist += diff * diff;
        }
        dist = std::sqrt(dist / (float)self_state.size());
        consistency = 1.0f / (1.0f + dist * 3.0f);

        float alpha = 0.01f;
        for (size_t i = 0; i < self_state.size(); i++) {
            self_reference[i] = self_reference[i] * (1.0f - alpha) + self_state[i] * alpha;
        }
    } else if (self_reference.empty() || self_reference.size() != self_state.size()) {
        self_reference = self_state;
        consistency = 0.5f;
    }

    float intrinsic_reward = novelty * 0.3f + progress * 0.4f + consistency * 0.3f;

    prev_error = prediction_error;
    return intrinsic_reward;
}