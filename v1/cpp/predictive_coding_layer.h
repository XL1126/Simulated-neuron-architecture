#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>

struct PCLayer {
    std::vector<float> representation;
    std::vector<float> prediction;
    std::vector<float> prediction_error;
    std::vector<float> top_down_weights;
    size_t n_units;
    size_t n_lower_units;
    bool is_top_level;
};

class PredictiveCodingNetwork {
public:
    PredictiveCodingNetwork();
    PredictiveCodingNetwork(size_t n_levels,
                             const std::vector<size_t>& units_per_level);

    void init(size_t n_levels, const std::vector<size_t>& units_per_level);
    void predict();
    void compute_errors();
    void update_representations(float lr, size_t n_iterations);
    void learn_weights(float lr);
    void set_sensory_input(const std::vector<float>& bottom_up);
    const std::vector<float>& get_representation(size_t level) const;
    const std::vector<float>& get_prediction_error(size_t level) const;
    std::vector<float> get_top_level_embedding() const;

    float compute_free_energy() const;
    void reset();

private:
    std::vector<PCLayer> layers;
    size_t n_levels;
    std::mt19937 rng;

    static constexpr float DEFAULT_WEIGHT_SCALE = 0.1f;
};