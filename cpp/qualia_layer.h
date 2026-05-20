#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct QualiaLayer {
    static constexpr int N_NEURONS = 128;
    static constexpr int SENSORY_DIM = 64;
    static constexpr int QUALIA_DIM = 32;

    std::vector<float> H_state;
    std::vector<float> P_sensory;
    std::vector<float> Q_readout;

    std::vector<float> qualia_vector;
    std::vector<float> bound_percept;
    std::vector<float> raw_feel;
    std::vector<float> last_sensory_input;
    float binding_strength;
    float perceptual_vividness;
    float first_person_salience;

    std::deque<std::vector<float>> qualia_history;
    static constexpr int MAX_QUALIA_HISTORY = 100;

    QualiaLayer();

    void init();

    void forward(const std::vector<float>& visual,
                  const std::vector<float>& auditory,
                  const std::vector<float>& tactile,
                  const std::vector<float>& vestibular,
                  const std::vector<float>& self_state,
                  float global_dopamine, float global_norepinephrine,
                  float dt_ms);

    void learn(float reward_signal);

    const std::vector<float>& get_qualia() const { return qualia_vector; }
    const std::vector<float>& get_bound_percept() const { return bound_percept; }
    float get_vividness() const { return perceptual_vividness; }
    float get_fp_salience() const { return first_person_salience; }
};