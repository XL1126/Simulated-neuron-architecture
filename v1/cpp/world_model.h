#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

class WorldModel {
public:
    static constexpr size_t SENSORY_DIM = 64;
    static constexpr size_t HIDDEN_DIM = 128;

    WorldModel();

    void init();

    std::vector<float> predict_next(const std::vector<float>& current_sensory);

    float compute_prediction_error(const std::vector<float>& predicted,
                                    const std::vector<float>& actual) const;

    void learn(const std::vector<float>& current_sensory,
               const std::vector<float>& next_sensory, float lr);

    float get_error() const { return world_prediction_error; }
    const std::vector<float>& get_last_prediction() const { return prediction; }

private:
    std::vector<float> W_enc;
    std::vector<float> b_enc;
    std::vector<float> W_pred;
    std::vector<float> b_pred;

    std::vector<float> encoded;
    std::vector<float> prediction;
    float world_prediction_error;
};

class CLUBEstimator {
public:
    static constexpr size_t SELF_DIM = 128;
    static constexpr size_t SENSORY_ENC_DIM = 64;
    static constexpr size_t CLUB_HIDDEN = 64;

    CLUBEstimator();

    void init();

    float estimate_mi(const std::vector<float>& self_state,
                       const std::vector<float>& sensory_encoding);

    float compute_mi_upper_bound(const std::vector<float>& self_batch,
                                   const std::vector<float>& sensory_batch,
                                   size_t batch_size);

    void update_estimator(const std::vector<float>& self_state,
                           const std::vector<float>& sensory_encoding, float lr);

    float get_mi_estimate() const { return current_mi; }

private:
    std::vector<float> W_self;
    std::vector<float> W_sensory;
    std::vector<float> b;

    float current_mi;

    std::vector<std::vector<float>> self_buffer;
    std::vector<std::vector<float>> sensory_buffer;
    size_t buffer_pos;
    static constexpr size_t BUFFER_SIZE = 32;

    float log_probability(const std::vector<float>& self,
                           const std::vector<float>& sensory) const;
};