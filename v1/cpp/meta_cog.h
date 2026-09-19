#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct MetaCogLayer {
    static constexpr int N_NEURONS = 200;
    static constexpr int SUMMARY_DIM = 64;

    std::vector<float> v;
    std::vector<float> u;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> I_syn;
    std::vector<uint8_t> fired;

    std::vector<float> W_input;
    std::vector<float> W_habits;

    float conf;
    float surp;
    float valence;
    float arousal;

    float prev_summary_norm;
    float prev_conf;
    float homeostasis_target;
    std::vector<float> prev_summary_vec;

    MetaCogLayer();

    void init();

    void forward(const std::vector<float>& summary, float world_error, float self_error, float dt_ms);

    void learn(const std::vector<float>& actual_summary, float reward_signal);

    float get_confidence() const { return conf; }
    float get_surprise() const { return surp; }
    float get_valence() const { return valence; }
    float get_arousal() const { return arousal; }
};