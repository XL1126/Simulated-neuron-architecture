#include "neuron_izhikevich.h"
#include <algorithm>

void izhikevich_init(NeuronState& n, uint8_t neuron_type, std::mt19937& rng) {
    n.neuron_type = neuron_type;
    n.fired = false;
    n.last_fire_step = 0;
    n.fire_count = 0;
    n.avg_firing_rate = 0.0f;
    n.I_syn = 0.0f;
    n.I_dendrite = 0.0f;
    n.I_noise = 0.0f;
    n.threshold = 30.0f;

    std::uniform_real_distribution<float> jitter(-2.0f, 2.0f);
    std::normal_distribution<float> bias_dist(3.5f, 1.2f);

    switch (neuron_type) {
        case NeuronState::TYPE_REGULAR:
            n.a = 0.02f; n.b = 0.2f;
            n.c = -65.0f + jitter(rng);
            n.d = 8.0f + jitter(rng) * 0.25f;
            n.I_bias = std::max(1.0f, bias_dist(rng));
            break;
        case NeuronState::TYPE_BURSTING:
            n.a = 0.02f; n.b = 0.25f;
            n.c = -55.0f + jitter(rng);
            n.d = 0.05f + jitter(rng) * 0.01f;
            n.I_bias = std::max(1.8f, bias_dist(rng) * 1.4f);
            break;
        case NeuronState::TYPE_FAST:
            n.a = 0.02f; n.b = 0.2f;
            n.c = -50.0f + jitter(rng);
            n.d = 2.0f + jitter(rng) * 0.25f;
            n.I_bias = std::max(2.5f, bias_dist(rng) * 1.8f);
            break;
        default:
            n.a = 0.02f; n.b = 0.2f;
            n.c = -65.0f; n.d = 8.0f;
            n.I_bias = 3.0f;
            break;
    }

    n.v = n.c;
    n.u = n.b * n.v;
}

void izhikevich_update(NeuronState& n, float noise_std) {
    n.fired = false;

    float I_total = n.I_syn + n.I_dendrite + n.I_bias + n.I_noise;

    n.v += 0.5f * (0.04f * n.v * n.v + 5.0f * n.v + 140.0f - n.u + I_total);
    n.v += 0.5f * (0.04f * n.v * n.v + 5.0f * n.v + 140.0f - n.u + I_total);
    n.u += n.a * (n.b * n.v - n.u);

    if (n.v >= n.threshold) {
        n.v = n.c;
        n.u = n.u + n.d;
        n.fired = true;
    }

    n.I_syn *= 0.8f;
    n.I_dendrite *= 0.95f;
    n.I_noise = 0.0f;
}

void izhikevich_update_threshold_drift(NeuronState& n, float dt_ms) {
    float delta = dt_ms / 1000.0f;
    if (n.avg_firing_rate < 0.5f) {
        n.threshold = std::max(22.0f, n.threshold - 0.02f * delta);
    } else if (n.avg_firing_rate > 8.0f) {
        n.threshold = std::min(32.0f, n.threshold + 0.03f * delta);
    } else {
        n.threshold += (30.0f - n.threshold) * 0.001f * delta;
    }
}
