#pragma once
#include <cstdint>
#include <vector>
#include <random>

struct NeuronState {
    float v;
    float u;
    float a;
    float b;
    float c;
    float d;
    float I_syn;
    float I_dendrite;
    float I_bias;
    float I_noise;
    float threshold;
    bool fired;
    uint64_t last_fire_step;
    uint32_t fire_count;
    float avg_firing_rate;
    uint32_t group_id;
    uint8_t neuron_type;

    float r1;
    float r2;
    float o1;
    float o2;

    static constexpr uint8_t TYPE_REGULAR = 0;
    static constexpr uint8_t TYPE_BURSTING = 1;
    static constexpr uint8_t TYPE_FAST = 2;
};

void izhikevich_init(NeuronState& n, uint8_t neuron_type, std::mt19937& rng);
void izhikevich_update(NeuronState& n, float noise_std);
void izhikevich_update_threshold_drift(NeuronState& n, float dt_ms);
