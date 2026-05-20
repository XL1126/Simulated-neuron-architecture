#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>

struct SPAMappingEntry {
    std::vector<uint32_t> neuron_ids;
    std::vector<float> base_weights;
    std::vector<float> eligibility;
    std::string concept;
    bool is_frozen;
};

class SPANeuralMapper {
public:
    SPANeuralMapper(uint32_t total_neurons, uint32_t dim,
                    uint32_t neurons_per_concept);

    std::vector<uint32_t> add_concept(const std::string& concept,
                                       bool force_new = false);
    std::vector<uint32_t> get_concept_neurons(const std::string& concept) const;

    void map_vector_to_spikes(const std::vector<float>& vec,
                               std::vector<std::pair<uint32_t, float>>& out_spikes,
                               float energy_scale = 1.0f);
    std::vector<float> map_spikes_to_vector(
        const std::vector<std::pair<uint32_t, float>>& active_spikes);

    void learn_from_pair(const std::vector<uint32_t>& pre_ids,
                         const std::vector<uint32_t>& post_ids,
                         float reward);

    void reinforce_concept(const std::string& concept, float strength);
    void decay_eligibility(float rate = 0.99f);

    uint32_t get_dim() const { return vector_dim; }
    uint32_t get_total_neurons() const { return total_neurons; }
    uint32_t get_neurons_per_concept() const { return neurons_per_concept; }

    std::vector<float> get_concept_vector(const std::string& concept) const;
    std::string resolve_neuron(uint32_t neuron_id) const;

private:
    uint32_t total_neurons;
    uint32_t vector_dim;
    uint32_t neurons_per_concept;

    std::vector<SPAMappingEntry> entries;
    std::unordered_map<std::string, uint32_t> concept_index;
    std::unordered_map<uint32_t, std::vector<std::string>> neuron_to_concepts;

    std::mt19937 rng;

    void _assign_neurons(SPAMappingEntry& entry, const std::string& concept);
    uint32_t _hash_str(const std::string& s) const;
    float _segment_energy(const std::vector<float>& vec,
                          uint32_t seg_idx, uint32_t total_segs) const;
};