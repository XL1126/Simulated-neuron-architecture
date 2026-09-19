#pragma once
#include "neuron_izhikevich.h"
#include "synapse.h"
#include "spike_event.h"
#include "dendrite_compartment.h"
#include "stdp_engine.h"
#include <vector>
#include <queue>
#include <deque>
#include <random>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct SpikeFireEvent {
    uint32_t neuron_id;
    float strength;
};

class NeuronPopulation {
public:
    NeuronPopulation(uint32_t num_neurons, uint32_t avg_degree);

    void update(uint64_t step, float noise_level, float dopamine);
    void apply_forgetting();
    void apply_stdp_consolidation();
    void apply_homeostasis(uint64_t step);
    void apply_sleep_cycle(uint64_t step);

    void inject_spike(uint32_t target_id, float strength, uint32_t delay_ms);
    void inject_spike_group(const std::vector<uint32_t>& target_ids, float strength, uint32_t delay_ms);

    void set_dopamine(float value);
    float get_dopamine() const;

    const std::vector<NeuronState>& get_neurons() const;
    std::vector<std::vector<uint32_t>> get_adjacency() const;
    const std::vector<SpikeFireEvent>& get_current_fires() const;

    void update_eligibility_traces();
    void apply_credit(float reward, float eta);
    void decay_eligibility_traces(float lambda);

    void set_neuron_bias(uint32_t neuron_id, float bias);
    void add_synaptic_current(uint32_t neuron_id, float current);

    void add_synapse(uint32_t src, uint32_t dst, float weight, uint8_t delay);
    void connect_random(float prob, uint32_t max_per_neuron);
    void build_small_world(uint32_t group_size, float local_prob, float long_range_prob);
    void build_competitive_pool(uint32_t start_idx, uint32_t end_idx);
    void build_erdos_renyi(uint32_t start_idx, uint32_t end_idx, float prob);

    size_t size() const { return neurons.size(); }

    void set_seed(uint64_t seed);

    void set_stdp_config(const STDPConfig& config);
    const STDPConfig& get_stdp_config() const;

    STDPEngine& get_stdp_engine() { return stdp_engine; }
    void process_triplet_stdp(uint64_t step, float td_error, float global_da, float global_ach);
    void update_triplet_variables(float dt_ms);
    void update_dendrite_calcium(float dt_ms);

    void set_td_state(float v, float v_prev) { td_v_current = v; td_v_previous = v_prev; }
    float compute_td_error(float reward, float gamma) const;

private:
    std::vector<NeuronState> neurons;
    std::vector<std::vector<Synapse>> out_synapses;
    std::vector<std::vector<FireHistoryEntry>> fire_history;
    std::vector<DendriteTree> dendrite_trees;
    std::deque<std::vector<SpikeEvent>> delay_queue;

    std::vector<SpikeFireEvent> current_fires;
    float global_dopamine;
    STDPConfig stdp_config;
    STDPEngine stdp_engine;

    float td_v_current;
    float td_v_previous;

    std::mt19937 rng;

    static constexpr uint32_t MAX_DELAY = 20;

    void process_delay_queue(uint64_t step);
    void process_stdp(uint64_t step);
    void update_stp();
    void record_firing(NeuronState& n, uint32_t neuron_id, uint64_t step);
    void setup_neurons_by_distribution();
    void setup_dendrite_trees(uint32_t num_compartments);
};
