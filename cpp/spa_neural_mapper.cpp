#include "spa_neural_mapper.h"
#include <cmath>
#include <algorithm>

SPANeuralMapper::SPANeuralMapper(uint32_t total_neurons_, uint32_t dim_,
                                   uint32_t neurons_per_concept_)
    : total_neurons(total_neurons_), vector_dim(dim_),
      neurons_per_concept(neurons_per_concept_)
{
    std::random_device rd;
    rng.seed(rd());
}

uint32_t SPANeuralMapper::_hash_str(const std::string& s) const {
    uint32_t h = 5381;
    for (char c : s) h = ((h << 5) + h) + (unsigned char)c;
    return h;
}

float SPANeuralMapper::_segment_energy(const std::vector<float>& vec,
                                        uint32_t seg_idx, uint32_t total_segs) const {
    if (vec.empty() || total_segs == 0) return 0.0f;
    uint32_t seg_size = (uint32_t)vec.size() / total_segs;
    if (seg_size == 0) seg_size = 1;
    uint32_t start = seg_idx * seg_size;
    uint32_t end = std::min(start + seg_size, (uint32_t)vec.size());
    float energy = 0.0f;
    for (uint32_t i = start; i < end; i++) energy += std::abs(vec[i]);
    return energy / (float)(end - start);
}

void SPANeuralMapper::_assign_neurons(SPAMappingEntry& entry,
                                       const std::string& concept) {
    uint32_t base = _hash_str(concept) % total_neurons;
    entry.neuron_ids.resize(neurons_per_concept);
    entry.base_weights.resize(neurons_per_concept);
    entry.eligibility.resize(neurons_per_concept);

    std::uniform_int_distribution<uint32_t> offset_dist(0, total_neurons / neurons_per_concept);

    for (uint32_t i = 0; i < neurons_per_concept; i++) {
        uint32_t nid = (base + i * 73 + offset_dist(rng)) % total_neurons;
        entry.neuron_ids[i] = nid;
        entry.base_weights[i] = 0.08f + (float)(i % 3) * 0.04f;
        entry.eligibility[i] = 0.1f;
        neuron_to_concepts[nid].push_back(concept);
    }
}

std::vector<uint32_t> SPANeuralMapper::add_concept(const std::string& concept,
                                                     bool force_new) {
    if (!force_new) {
        auto it = concept_index.find(concept);
        if (it != concept_index.end()) return entries[it->second].neuron_ids;
    }

    SPAMappingEntry entry;
    entry.concept = concept;
    entry.is_frozen = false;
    _assign_neurons(entry, concept);

    uint32_t idx = (uint32_t)entries.size();
    entries.push_back(entry);
    concept_index[concept] = idx;

    return entries[idx].neuron_ids;
}

std::vector<uint32_t> SPANeuralMapper::get_concept_neurons(const std::string& concept) const {
    auto it = concept_index.find(concept);
    if (it != concept_index.end()) return entries[it->second].neuron_ids;
    return {};
}

void SPANeuralMapper::map_vector_to_spikes(
    const std::vector<float>& vec,
    std::vector<std::pair<uint32_t, float>>& out_spikes,
    float energy_scale)
{
    out_spikes.clear();
    if (vec.empty()) return;

    float total_energy = 0.0f;
    std::vector<float> seg_energies(neurons_per_concept, 0.0f);
    for (uint32_t i = 0; i < neurons_per_concept; i++) {
        seg_energies[i] = _segment_energy(vec, i, neurons_per_concept);
        total_energy += seg_energies[i];
    }

    float norm = (total_energy > 0.0001f) ? (energy_scale / total_energy) : 1.0f;

    for (auto& entry : entries) {
        float concept_match = 0.0f;
        if (!entry.concept.empty()) {
            std::vector<float> cvec = get_concept_vector(entry.concept);
            float dot = 0.0f;
            for (size_t i = 0; i < std::min(vec.size(), cvec.size()); i++)
                dot += vec[i] * cvec[i];
            concept_match = std::abs(dot) * 2.0f;
        }

        for (uint32_t i = 0; i < entry.neuron_ids.size(); i++) {
            float strength = seg_energies[i] * norm * entry.base_weights[i];
            if (concept_match > 0.3f) strength += concept_match * 0.5f * entry.eligibility[i];
            if (strength > 0.01f) {
                out_spikes.push_back({entry.neuron_ids[i],
                    std::min(2.0f, strength)});
            }
        }
    }

    std::sort(out_spikes.begin(), out_spikes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (out_spikes.size() > 200) out_spikes.resize(200);
}

std::vector<float> SPANeuralMapper::map_spikes_to_vector(
    const std::vector<std::pair<uint32_t, float>>& active_spikes)
{
    std::vector<float> result(vector_dim, 0.0f);

    for (auto& [nid, strength] : active_spikes) {
        auto it = neuron_to_concepts.find(nid);
        if (it == neuron_to_concepts.end()) continue;

        for (auto& concept : it->second) {
            auto idx_it = concept_index.find(concept);
            if (idx_it == concept_index.end()) continue;
            auto& entry = entries[idx_it->second];

            for (uint32_t i = 0; i < entry.neuron_ids.size(); i++) {
                if (entry.neuron_ids[i] == nid) {
                    uint32_t seg_size = vector_dim / neurons_per_concept;
                    if (seg_size == 0) seg_size = 1;
                    uint32_t start = i * seg_size;
                    uint32_t end = std::min(start + seg_size, vector_dim);
                    float val = strength * entry.eligibility[i];
                    for (uint32_t j = start; j < end; j++) {
                        result[j] += val * ((float)(j - start + 1) / (float)seg_size);
                    }
                }
            }
        }
    }

    float norm = 0.0f;
    for (auto v : result) norm += v * v;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& v : result) v /= norm;
    }
    return result;
}

void SPANeuralMapper::learn_from_pair(const std::vector<uint32_t>& pre_ids,
                                       const std::vector<uint32_t>& post_ids,
                                       float reward)
{
    if (reward == 0.0f) return;

    for (auto pre_nid : pre_ids) {
        auto pre_it = neuron_to_concepts.find(pre_nid);
        if (pre_it == neuron_to_concepts.end()) continue;

        for (auto post_nid : post_ids) {
            auto post_it = neuron_to_concepts.find(post_nid);
            if (post_it == neuron_to_concepts.end()) continue;

            for (auto& pre_c : pre_it->second) {
                auto pre_idx = concept_index.find(pre_c);
                if (pre_idx == concept_index.end()) continue;

                for (uint32_t i = 0; i < entries[pre_idx->second].neuron_ids.size(); i++) {
                    if (entries[pre_idx->second].neuron_ids[i] == pre_nid) {
                        entries[pre_idx->second].eligibility[i] += reward * 0.01f;
                        entries[pre_idx->second].eligibility[i] =
                            std::max(0.001f, std::min(1.0f,
                                entries[pre_idx->second].eligibility[i]));
                    }
                }
            }
        }
    }
}

void SPANeuralMapper::reinforce_concept(const std::string& concept, float strength) {
    auto it = concept_index.find(concept);
    if (it == concept_index.end()) return;
    auto& entry = entries[it->second];
    for (auto& e : entry.eligibility) {
        e += strength * 0.02f;
        e = std::max(0.001f, std::min(1.0f, e));
    }
}

void SPANeuralMapper::decay_eligibility(float rate) {
    for (auto& entry : entries) {
        if (entry.is_frozen) continue;
        for (auto& e : entry.eligibility) e *= rate;
    }
}

std::vector<float> SPANeuralMapper::get_concept_vector(const std::string& concept) const {
    auto it = concept_index.find(concept);
    std::vector<float> result(vector_dim, 0.0f);
    if (it == concept_index.end()) return result;

    auto& entry = entries[it->second];
    uint32_t seg_size = vector_dim / neurons_per_concept;
    if (seg_size == 0) seg_size = 1;

    for (uint32_t i = 0; i < entry.neuron_ids.size(); i++) {
        uint32_t start = i * seg_size;
        uint32_t end = std::min(start + seg_size, vector_dim);
        float val = entry.eligibility[i] * (1.0f + (float)(i % 3) * 0.1f);
        for (uint32_t j = start; j < end; j++) result[j] = val;
    }

    float norm = 0.0f;
    for (auto v : result) norm += v * v;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& v : result) v /= norm;
    }
    return result;
}

std::string SPANeuralMapper::resolve_neuron(uint32_t neuron_id) const {
    auto it = neuron_to_concepts.find(neuron_id);
    if (it != neuron_to_concepts.end() && !it->second.empty())
        return it->second[0];
    return "";
}