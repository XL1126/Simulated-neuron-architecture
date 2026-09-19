#pragma once
#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

struct DendriteCompartment {
    float coeff;
    std::deque<float> input_window;
    uint32_t window_size;
    float sigmoid_steepness;

    float ca_concentration;
    float nmda_conductance;
    float l_type_vgcc_conductance;
    float distance_from_soma;

    static constexpr float CA_BASELINE = 0.1f;
    static constexpr float TAU_CA_BASE = 50.0f;
    static constexpr float KAPPA_NMDA = 0.3f;
    static constexpr float KAPPA_VGCC = 0.5f;

    DendriteCompartment() : coeff(0.5f), window_size(10),
        sigmoid_steepness(1.0f), ca_concentration(CA_BASELINE),
        nmda_conductance(0.1f), l_type_vgcc_conductance(0.05f),
        distance_from_soma(100.0f) {}

    float tau_ca_decay() const { return TAU_CA_BASE + 0.5f * distance_from_soma; }

    float compute_output() const;
    void add_input(float strength);
    void decay_window();

    void update_calcium(float vm, float dt_ms) {
        float tau = tau_ca_decay();
        float decay = -(ca_concentration - CA_BASELINE) / tau * dt_ms;
        float nmda_influx = KAPPA_NMDA * nmda_conductance * std::max(0.0f, vm + 70.0f) / 100.0f;
        nmda_influx = std::max(0.0f, std::min(1.0f, nmda_influx));
        float vgcc_thresh = -30.0f;
        float vgcc_influx = (vm > vgcc_thresh)
            ? KAPPA_VGCC * l_type_vgcc_conductance * (vm - vgcc_thresh) / 100.0f : 0.0f;
        vgcc_influx = std::max(0.0f, std::min(1.0f, vgcc_influx));

        ca_concentration += decay * dt_ms + nmda_influx * dt_ms * 0.01f + vgcc_influx * dt_ms * 0.005f;
        if (ca_concentration < 0.0f) ca_concentration = 0.0f;
        if (!std::isfinite(ca_concentration)) ca_concentration = CA_BASELINE;
    }

    float calc_calcium_gated_plasticity() const {
        float threshold_ltd = 0.5f;
        float threshold_ltp = 1.0f;
        if (ca_concentration < threshold_ltd) return 0.0f;
        if (ca_concentration < threshold_ltp)
            return -(ca_concentration - threshold_ltd) / (threshold_ltp - threshold_ltd);
        return std::min(1.0f, (ca_concentration - threshold_ltp) / 2.0f);
    }
};

struct DendriteTree {
    std::vector<DendriteCompartment> compartments;
    void init(uint32_t num_compartments, uint32_t window_size);
    float compute_total_output();
    void add_input_distributed(float strength);
    void step();
    void update_calcium_on_all(float vm, float dt_ms);
};