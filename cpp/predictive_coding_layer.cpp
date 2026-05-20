#include "predictive_coding_layer.h"
#include <cmath>

static inline bool is_bad(float x) {
    return !std::isfinite(x);
}

PredictiveCodingNetwork::PredictiveCodingNetwork()
    : n_levels(0) {
    rng.seed(77);
}

PredictiveCodingNetwork::PredictiveCodingNetwork(
    size_t n_levels, const std::vector<size_t>& units_per_level)
    : n_levels(n_levels) {
    rng.seed(77);
    layers.resize(n_levels);
    for (size_t l = 0; l < n_levels; l++) {
        auto& layer = layers[l];
        layer.n_units = units_per_level[l];
        layer.n_lower_units = (l > 0) ? units_per_level[l - 1] : 0;
        layer.is_top_level = (l == n_levels - 1);
        layer.representation.resize(layer.n_units, 0.01f);
        layer.prediction.resize((l < n_levels - 1) ? layer.n_units : 0, 0.0f);
        layer.prediction_error.resize(layer.n_units, 0.0f);

        if (l > 0) {
            layer.top_down_weights.resize(layer.n_units * layer.n_lower_units, 0.0f);
            for (size_t i = 0; i < layer.n_units; i++) {
                for (size_t j = 0; j < layer.n_lower_units; j++) {
                    uint32_t h = (uint32_t)(i * 100003 + j * 99991 + l * 7777);
                    layer.top_down_weights[i * layer.n_lower_units + j] =
                        ((float)(h % 20000) / 10000.0f - 1.0f) * DEFAULT_WEIGHT_SCALE;
                }
            }
        }
    }
}

void PredictiveCodingNetwork::init(size_t n_levels,
                                    const std::vector<size_t>& units_per_level) {
    this->n_levels = n_levels;
    layers.resize(n_levels);
    for (size_t l = 0; l < n_levels; l++) {
        auto& layer = layers[l];
        layer.n_units = units_per_level[l];
        layer.n_lower_units = (l > 0) ? units_per_level[l - 1] : 0;
        layer.is_top_level = (l == n_levels - 1);
        layer.representation.resize(layer.n_units, 0.01f);
        layer.prediction.resize((l < n_levels - 1) ? layer.n_units : 0, 0.0f);
        layer.prediction_error.resize(layer.n_units, 0.0f);

        if (l > 0) {
            layer.top_down_weights.resize(layer.n_units * layer.n_lower_units, 0.0f);
            for (size_t i = 0; i < layer.n_units; i++) {
                for (size_t j = 0; j < layer.n_lower_units; j++) {
                    uint32_t h = (uint32_t)(i * 100003 + j * 99991 + l * 7777);
                    layer.top_down_weights[i * layer.n_lower_units + j] =
                        ((float)(h % 20000) / 10000.0f - 1.0f) * DEFAULT_WEIGHT_SCALE;
                }
            }
        }
    }
}

void PredictiveCodingNetwork::predict() {
    for (size_t l = 1; l < n_levels; l++) {
        auto& top = layers[l];
        auto& bottom = layers[l - 1];
        if (top.n_lower_units == 0) continue;
        for (size_t j = 0; j < bottom.n_units; j++) {
            float pred = 0.0f;
            for (size_t i = 0; i < top.n_units; i++) {
                float r = top.representation[i];
                float w = top.top_down_weights[i * top.n_lower_units + j];
                if (is_bad(r)) r = 0.0f;
                if (is_bad(w)) w = 0.0f;
                pred += r * w;
            }
            bottom.prediction[j] = is_bad(pred) ? 0.0f : std::tanh(pred);
        }
    }
}

void PredictiveCodingNetwork::compute_errors() {
    for (size_t l = 0; l < n_levels; l++) {
        auto& layer = layers[l];
        for (size_t i = 0; i < layer.n_units; i++) {
            layer.prediction_error[i] = is_bad(layer.representation[i])
                ? 0.0f : layer.representation[i];
        }

        if (l > 0 && l - 1 < layers.size()) {
            auto& below = layers[l - 1];
            for (size_t i = 0; i < layer.n_units; i++) {
                float bottom_up_err = 0.0f;
                size_t limit = std::min(below.prediction_error.size(), layer.n_lower_units);
                for (size_t j = 0; j < limit; j++) {
                    size_t widx = i * layer.n_lower_units + j;
                    if (widx < layer.top_down_weights.size()) {
                        float be = below.prediction_error[j];
                        float tw = layer.top_down_weights[widx];
                        if (is_bad(be)) be = 0.0f;
                        if (is_bad(tw)) tw = 0.0f;
                        bottom_up_err += be * tw;
                    }
                }
                float& pe = layer.prediction_error[i];
                pe -= bottom_up_err * 0.3f;
                if (is_bad(pe)) pe = 0.0f;
            }
        }
    }
}

void PredictiveCodingNetwork::update_representations(float lr, size_t n_iterations) {
    for (size_t iter = 0; iter < n_iterations; iter++) {
        predict();
        compute_errors();
        for (size_t l = 0; l < n_levels; l++) {
            auto& layer = layers[l];
            for (size_t i = 0; i < layer.n_units; i++) {
                float grad = layer.prediction_error[i];
                if (is_bad(grad)) grad = 0.0f;
                if (l < n_levels - 1) {
                    float extra = layer.prediction_error[i] * 0.5f;
                    grad += is_bad(extra) ? 0.0f : extra;
                }
                layer.representation[i] += lr * grad;
                float& rep = layer.representation[i];
                if (is_bad(rep)) rep = 0.0f;
                if (rep > 1.0f) rep = 1.0f;
                else if (rep < -1.0f) rep = -1.0f;
            }
        }
    }
}

void PredictiveCodingNetwork::learn_weights(float lr) {
    for (size_t l = 1; l < n_levels; l++) {
        auto& layer = layers[l];
        auto& below = layers[l - 1];
        if (layer.n_lower_units == 0) continue;
        for (size_t i = 0; i < layer.n_units; i++) {
            float err_i = layer.prediction_error[i];
            if (is_bad(err_i)) continue;
            for (size_t j = 0; j < std::min(below.n_units, layer.n_lower_units); j++) {
                float rep_j = below.representation[j];
                if (is_bad(rep_j)) continue;
                float delta = lr * err_i * rep_j;
                size_t idx = i * layer.n_lower_units + j;
                float& w = layer.top_down_weights[idx];
                w += delta;
                if (is_bad(w)) w = 0.0f;
                if (w > 0.5f) w = 0.5f;
                else if (w < -0.5f) w = -0.5f;
            }
        }
    }
}

void PredictiveCodingNetwork::set_sensory_input(
    const std::vector<float>& bottom_up) {
    if (layers.empty()) return;
    auto& bottom = layers[0];
    for (size_t i = 0; i < std::min(bottom.n_units, bottom_up.size()); i++) {
        float val = bottom_up[i] * 2.0f - 1.0f;
        if (is_bad(val)) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        else if (val < -1.0f) val = -1.0f;
        bottom.representation[i] = val;
    }
}

const std::vector<float>& PredictiveCodingNetwork::get_representation(
    size_t level) const {
    return layers[level].representation;
}

const std::vector<float>& PredictiveCodingNetwork::get_prediction_error(
    size_t level) const {
    return layers[level].prediction_error;
}

std::vector<float> PredictiveCodingNetwork::get_top_level_embedding() const {
    if (layers.empty()) return {};
    return layers.back().representation;
}

float PredictiveCodingNetwork::compute_free_energy() const {
    float fe = 0.0f;
    for (auto& layer : layers) {
        for (auto err : layer.prediction_error) {
            fe += err * err;
        }
    }
    return fe / (float)layers.size();
}

void PredictiveCodingNetwork::reset() {
    for (auto& layer : layers) {
        std::fill(layer.representation.begin(), layer.representation.end(), 0.01f);
        std::fill(layer.prediction.begin(), layer.prediction.end(), 0.0f);
        std::fill(layer.prediction_error.begin(), layer.prediction_error.end(), 0.0f);
    }
}