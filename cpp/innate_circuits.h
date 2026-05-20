#pragma once
#include <cstdint>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <unordered_map>
#include <string>

struct InbornSynapse {
    uint32_t src;
    uint32_t dst;
    float weight;
    uint8_t delay;
    bool is_core;
};

class InnateCircuitBuilder {
public:
    InnateCircuitBuilder(uint32_t seed = 42);

    std::vector<InbornSynapse>
    build_visual_retinotopic(uint32_t base_id, uint32_t n_neurons,
                              uint32_t field_w, uint32_t field_h);
    std::vector<InbornSynapse>
    build_visual_orientation_columns(uint32_t base_id, uint32_t n_neurons,
                                      uint32_t n_orientations);
    std::vector<InbornSynapse>
    build_motor_somatotopic(uint32_t base_id, uint32_t n_neurons,
                             uint32_t n_actions, uint32_t n_directions);
    std::vector<InbornSynapse>
    build_hippocampal_dg_ca3_ca1(uint32_t dg_base, uint32_t dg_n,
                                   uint32_t ca3_base, uint32_t ca3_n,
                                   uint32_t ca1_base, uint32_t ca1_n);
    std::vector<InbornSynapse>
    build_prefrontal_working_memory(uint32_t base_id, uint32_t n_neurons,
                                     uint32_t n_slots);
    std::vector<InbornSynapse>
    build_amygdala_valence(uint32_t base_id, uint32_t n_neurons);
    std::vector<InbornSynapse>
    build_language_semantic(uint32_t base_id, uint32_t n_neurons,
                             const std::vector<std::string>& concepts);
    std::vector<InbornSynapse>
    build_global_workspace(uint32_t base_id, uint32_t n_neurons);
    std::vector<InbornSynapse>
    build_inter_region(uint32_t src_base, uint32_t src_n,
                       uint32_t dst_base, uint32_t dst_n,
                       float prob, float max_weight, uint8_t max_delay);

    std::vector<InbornSynapse>
    build_thalamic_relay(uint32_t base_id, uint32_t n_neurons);

    std::vector<InbornSynapse>
    build_claustrum_coordinator(uint32_t base_id, uint32_t n_neurons);

    std::vector<InbornSynapse>
    build_default_mode_network(uint32_t base_id, uint32_t n_neurons);

    std::vector<InbornSynapse>
    build_workspace_three_layer(uint32_t base_id, uint32_t n_total,
                                 uint32_t n_sensory, uint32_t n_competition,
                                 uint32_t n_broadcast);

    std::vector<InbornSynapse>
    build_hippocampal_structured(uint32_t dg_base, uint32_t dg_n,
                                   uint32_t ca3_base, uint32_t ca3_n,
                                   uint32_t ca1_base, uint32_t ca1_n,
                                   float dg_sparsity, float ca3_recurrent_prob);

    std::vector<InbornSynapse>
    build_visual_hierarchical(uint32_t v1_base, uint32_t v1_n,
                               uint32_t v2_base, uint32_t v2_n,
                               uint32_t n_orientations);

    std::vector<InbornSynapse>
    build_prefrontal_esn(uint32_t base_id, uint32_t n_neurons,
                          uint32_t reservoir_size, uint32_t readout_size);

private:
    std::mt19937 rng;
    float _gabor(float x, float y, float theta, float sigma, float freq, float phase);
    float _dog(float x, float y, float sigma_c, float sigma_s);
};