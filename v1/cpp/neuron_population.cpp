#include "neuron_population.h"
#include <stdexcept>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

NeuronPopulation::NeuronPopulation(uint32_t num_neurons, uint32_t avg_degree)
    : global_dopamine(0.5f), td_v_current(0.0f), td_v_previous(0.0f)
{
    std::random_device rd;
    rng.seed(rd());

    neurons.resize(num_neurons);
    out_synapses.resize(num_neurons);
    fire_history.resize(num_neurons);
    dendrite_trees.resize(num_neurons);
    delay_queue.resize(MAX_DELAY + 1);

    setup_neurons_by_distribution();
    setup_dendrite_trees(2);

    stdp_engine.mu_d = 0.4f;
    stdp_engine.mu_a = 0.2f;
    stdp_engine.lr_global = 0.01f;
}

void NeuronPopulation::setup_neurons_by_distribution() {
    size_t n = neurons.size();
    size_t regular_count = (size_t)(n * 0.6);
    size_t bursting_count = (size_t)(n * 0.3);
    size_t fast_count = n - regular_count - bursting_count;

    for (size_t i = 0; i < regular_count; i++) {
        izhikevich_init(neurons[i], NeuronState::TYPE_REGULAR, rng);
        neurons[i].group_id = (uint32_t)(i % 256);
    }
    for (size_t i = regular_count; i < regular_count + bursting_count; i++) {
        izhikevich_init(neurons[i], NeuronState::TYPE_BURSTING, rng);
        neurons[i].group_id = (uint32_t)(i % 256);
    }
    for (size_t i = regular_count + bursting_count; i < n; i++) {
        izhikevich_init(neurons[i], NeuronState::TYPE_FAST, rng);
        neurons[i].group_id = (uint32_t)(i % 256);
    }
}

void NeuronPopulation::setup_dendrite_trees(uint32_t num_compartments) {
    for (auto& dt : dendrite_trees) {
        dt.init(num_compartments, 10);
    }
}

void NeuronPopulation::update(uint64_t step, float noise_level, float dopamine) {
    global_dopamine = dopamine;
    current_fires.clear();

    process_delay_queue(step);

    struct FireInfo {
        uint32_t neuron_id;
        float fire_strength;
        bool fired;
    };
    std::vector<FireInfo> fire_infos(neurons.size());

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (intptr_t i = 0; i < (intptr_t)neurons.size(); i++) {
        neurons[i].I_noise = std::normal_distribution<float>(0.0f, noise_level)(rng);
        neurons[i].I_dendrite = dendrite_trees[i].compute_total_output();
        izhikevich_update(neurons[i], noise_level);

        fire_infos[i].neuron_id = (uint32_t)i;
        fire_infos[i].fired = neurons[i].fired;
        if (neurons[i].fired) {
            float overshoot = (neurons[i].v - neurons[i].threshold) / 10.0f + 0.5f;
            fire_infos[i].fire_strength = std::min(2.0f, std::max(0.1f, 0.5f + overshoot));
        } else {
            fire_infos[i].fire_strength = 0.0f;
        }
    }

    for (size_t i = 0; i < neurons.size(); i++) {
        if (!fire_infos[i].fired) continue;
        record_firing(neurons[i], (uint32_t)i, step);
        current_fires.push_back({(uint32_t)i, fire_infos[i].fire_strength});

        for (auto& syn : out_synapses[i]) {
            syn.update_stp_on_spike();
            float effective_strength = syn.weight * syn.compute_stp_effect();

            SpikeEvent evt;
            evt.src_id = (uint32_t)i;
            evt.dst_id = syn.target_id;
            evt.strength = effective_strength;
            evt.delay_ms = syn.delay;
            evt.time_step = step + syn.delay;

            uint32_t delay_idx = syn.delay;
            if (delay_idx < delay_queue.size()) {
                delay_queue[delay_idx].push_back(evt);
                syn.last_use_step = step;
            }

            FireHistoryEntry fhe;
            fhe.time_step = step;
            fhe.src_id = (uint32_t)i;
            fhe.dst_id = syn.target_id;
            fhe.strength = effective_strength;
            fire_history[syn.target_id].push_back(fhe);
            if (fire_history[syn.target_id].size() > 200) {
                fire_history[syn.target_id].erase(fire_history[syn.target_id].begin());
            }
        }
    }

    process_stdp(step);
    update_stp();
    update_triplet_variables(1.0f);
    update_dendrite_calcium(1.0f);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (intptr_t i = 0; i < (intptr_t)dendrite_trees.size(); i++) {
        dendrite_trees[i].step();
    }

    for (auto& n : neurons) {
        if (n.fire_count > 0 && step > 0) {
            n.avg_firing_rate = (float)n.fire_count / (float)step * 1000.0f;
        }
    }
}

void NeuronPopulation::process_delay_queue(uint64_t step) {
    auto& current_spikes = delay_queue[0];
    for (auto& evt : current_spikes) {
        if (evt.dst_id < neurons.size()) {
            neurons[evt.dst_id].I_syn += evt.strength;
            dendrite_trees[evt.dst_id].add_input_distributed(evt.strength);
        }
    }
    current_spikes.clear();
    delay_queue.pop_front();
    delay_queue.push_back(std::vector<SpikeEvent>());
}

void NeuronPopulation::process_stdp(uint64_t step) {
    for (size_t post_id = 0; post_id < neurons.size(); post_id++) {
        auto& history = fire_history[post_id];
        for (size_t h = 0; h < history.size(); h++) {
            auto& fhe = history[h];
            uint64_t t_pre = fhe.time_step;
            if (step - t_pre > stdp_config.history_window_ms) continue;

            if (neurons[post_id].fired && neurons[post_id].last_fire_step > t_pre) {
                for (auto& syn : out_synapses[fhe.src_id]) {
                    if (syn.target_id == post_id) {
                        stdp_update_synapse(syn, t_pre, neurons[post_id].last_fire_step,
                                           global_dopamine, stdp_config);
                        stdp_update_eligibility(syn, t_pre, neurons[post_id].last_fire_step, 0.9f);
                    }
                }
            }
        }
    }
}

void NeuronPopulation::update_stp() {
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) syn.decay_stp(1.0f);
    }
}

void NeuronPopulation::record_firing(NeuronState& n, uint32_t neuron_id, uint64_t step) {
    n.last_fire_step = step;
    n.fire_count++;
}

void NeuronPopulation::apply_forgetting() {
    for (auto& syn_list : out_synapses) {
        syn_list.erase(
            std::remove_if(syn_list.begin(), syn_list.end(),
                [](const Synapse& s) { return !s.is_core && s.weight < 0.01f; }),
            syn_list.end()
        );
        for (auto& syn : syn_list) {
            if (!syn.is_core) syn.weight *= 0.95f;
        }
    }
}

void NeuronPopulation::apply_stdp_consolidation() {
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) {
            if (syn.weight > 0.8f) syn.is_core = true;
        }
    }
}

void NeuronPopulation::apply_homeostasis(uint64_t step) {
    float target_rate = 2.0f;
    float strength = 0.01f;

    for (size_t i = 0; i < neurons.size(); i++) {
        float deviation = neurons[i].avg_firing_rate - target_rate;
        if (std::abs(deviation) < 0.1f) continue;

        for (auto& syn_list : out_synapses) {
            for (auto& syn : syn_list) {
                if (syn.target_id == (uint32_t)i) {
                    syn.weight -= strength * deviation * 0.1f;
                    if (syn.weight < 0.0f) syn.weight = 0.0f;
                    if (syn.weight > 1.0f) syn.weight = 1.0f;
                }
            }
        }
        izhikevich_update_threshold_drift(neurons[i], 1.0f);
    }
}

void NeuronPopulation::apply_sleep_cycle(uint64_t step) {
    apply_homeostasis(step);
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) {
            if (syn.is_core) {
                syn.weight *= 1.01f;
                if (syn.weight > 1.0f) syn.weight = 1.0f;
            }
        }
    }
    apply_stdp_consolidation();
    apply_forgetting();
}

void NeuronPopulation::inject_spike(uint32_t target_id, float strength, uint32_t delay_ms) {
    if (target_id >= neurons.size()) return;
    SpikeEvent evt;
    evt.src_id = 0xFFFFFFFF;
    evt.dst_id = target_id;
    evt.strength = strength;
    evt.delay_ms = (uint16_t)delay_ms;
    evt.time_step = 1;
    uint32_t idx = std::min(delay_ms, MAX_DELAY);
    delay_queue[idx].push_back(evt);
}

void NeuronPopulation::inject_spike_group(const std::vector<uint32_t>& target_ids, float strength, uint32_t delay_ms) {
    for (auto id : target_ids) inject_spike(id, strength, delay_ms);
}

void NeuronPopulation::set_dopamine(float value) {
    global_dopamine = std::max(0.0f, std::min(1.0f, value));
}

float NeuronPopulation::get_dopamine() const { return global_dopamine; }

const std::vector<NeuronState>& NeuronPopulation::get_neurons() const { return neurons; }

std::vector<std::vector<uint32_t>> NeuronPopulation::get_adjacency() const {
    std::vector<std::vector<uint32_t>> adj(neurons.size());
    for (size_t i = 0; i < out_synapses.size(); i++) {
        for (auto& syn : out_synapses[i]) adj[i].push_back(syn.target_id);
    }
    return adj;
}

const std::vector<SpikeFireEvent>& NeuronPopulation::get_current_fires() const { return current_fires; }

void NeuronPopulation::update_eligibility_traces() {
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) syn.eligibility_trace *= 0.9f;
    }
}

void NeuronPopulation::apply_credit(float reward, float eta) {
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) stdp_apply_credit(syn, reward, eta);
    }
}

void NeuronPopulation::decay_eligibility_traces(float lambda) {
    for (auto& syn_list : out_synapses) {
        for (auto& syn : syn_list) syn.eligibility_trace *= lambda;
    }
}

void NeuronPopulation::set_neuron_bias(uint32_t neuron_id, float bias) {
    if (neuron_id < neurons.size()) neurons[neuron_id].I_bias = bias;
}

void NeuronPopulation::add_synaptic_current(uint32_t neuron_id, float current) {
    if (neuron_id < neurons.size()) neurons[neuron_id].I_syn += current;
}

void NeuronPopulation::add_synapse(uint32_t src, uint32_t dst, float weight, uint8_t delay) {
    if (src >= neurons.size() || dst >= neurons.size()) return;
    for (auto& syn : out_synapses[src]) {
        if (syn.target_id == dst) { syn.weight = weight; syn.delay = delay; return; }
    }
    Synapse syn;
    syn.target_id = dst; syn.weight = weight; syn.delay = delay;
    syn.d1_density = 0.5f + 0.5f * std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    syn.d2_density = 0.5f + 0.5f * std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    syn.d1_density = std::max(0.1f, std::min(1.0f, syn.d1_density));
    syn.d2_density = std::max(0.1f, std::min(1.0f, syn.d2_density));
    out_synapses[src].push_back(syn);
}

void NeuronPopulation::connect_random(float prob, uint32_t max_per_neuron) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < neurons.size(); i++) {
        uint32_t added = 0;
        for (size_t j = 0; j < neurons.size() && added < max_per_neuron; j++) {
            if (i != j && dist(rng) < prob) {
                Synapse syn;
                syn.target_id = (uint32_t)j;
                syn.weight = 0.05f + dist(rng) * 0.1f;
                syn.delay = (uint8_t)(1 + (int)(dist(rng) * 5));
                out_synapses[i].push_back(syn);
                added++;
            }
        }
    }
}

void NeuronPopulation::build_small_world(uint32_t group_size, float local_prob, float long_range_prob) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    uint32_t num_groups = (uint32_t)neurons.size() / group_size;

    for (uint32_t g = 0; g < num_groups; g++) {
        uint32_t start = g * group_size;
        uint32_t end = std::min(start + group_size, (uint32_t)neurons.size());
        for (uint32_t i = start; i < end; i++) {
            for (uint32_t j = start; j < end; j++) {
                if (i != j && dist(rng) < local_prob) {
                    add_synapse(i, j, 0.05f + dist(rng) * 0.1f, (uint8_t)(1 + (int)(dist(rng) * 3)));
                }
            }
        }
    }

    for (uint32_t i = 0; i < neurons.size(); i++) {
        if (dist(rng) < long_range_prob) {
            uint32_t j = (uint32_t)(dist(rng) * neurons.size());
            if (j != i && j < neurons.size()) {
                add_synapse(i, j, 0.01f + dist(rng) * 0.05f, (uint8_t)(1 + (int)(dist(rng) * 10)));
            }
        }
    }
}

void NeuronPopulation::build_competitive_pool(uint32_t start_idx, uint32_t end_idx) {
    start_idx = std::min(start_idx, (uint32_t)neurons.size());
    end_idx = std::min(end_idx, (uint32_t)neurons.size());
    for (uint32_t i = start_idx; i < end_idx; i++) {
        for (uint32_t j = start_idx; j < end_idx; j++) {
            add_synapse(i, j, (i == j) ? 1.0f : -0.5f, 1);
        }
    }
}

void NeuronPopulation::build_erdos_renyi(uint32_t start_idx, uint32_t end_idx, float prob) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    start_idx = std::min(start_idx, (uint32_t)neurons.size());
    end_idx = std::min(end_idx, (uint32_t)neurons.size());
    for (uint32_t i = start_idx; i < end_idx; i++) {
        for (uint32_t j = start_idx; j < end_idx; j++) {
            if (i != j && dist(rng) < prob) {
                add_synapse(i, j, 0.05f + dist(rng) * 0.1f, (uint8_t)(1 + (int)(dist(rng) * 5)));
            }
        }
    }
}

void NeuronPopulation::set_seed(uint64_t seed) {
    rng.seed((unsigned int)seed);
}

void NeuronPopulation::set_stdp_config(const STDPConfig& config) { stdp_config = config; }
const STDPConfig& NeuronPopulation::get_stdp_config() const { return stdp_config; }

void NeuronPopulation::update_triplet_variables(float dt_ms) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (intptr_t i = 0; i < (intptr_t)neurons.size(); i++) {
        stdp_engine.decay_triplet_variables(neurons[i], dt_ms);
    }
}

void NeuronPopulation::update_dendrite_calcium(float dt_ms) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (intptr_t i = 0; i < (intptr_t)neurons.size(); i++) {
        dendrite_trees[i].update_calcium_on_all(neurons[i].v, dt_ms);
    }
}

void NeuronPopulation::process_triplet_stdp(uint64_t step, float td_error,
                                              float global_da, float global_ach) {
    float mod_signal = stdp_engine.compute_modulation_factor(
        global_da, global_ach, 0.3f, td_error);

    for (size_t post_id = 0; post_id < neurons.size(); post_id++) {
        auto& history = fire_history[post_id];
        for (size_t h = 0; h < history.size(); h++) {
            auto& fhe = history[h];
            uint64_t t_pre = fhe.time_step;
            if (step - t_pre > stdp_config.history_window_ms) continue;

            uint32_t pre_id = fhe.src_id;
            if (pre_id >= neurons.size()) continue;

            if (neurons[post_id].fired && neurons[post_id].last_fire_step > t_pre) {
                for (auto& syn : out_synapses[pre_id]) {
                    if (syn.target_id == post_id) {
                        float da_factor = stdp_engine.compute_dopamine_factor(syn, global_da);
                        float combined_mod = mod_signal * (0.5f + da_factor);

                        stdp_engine.apply_triplet_three_factor(
                            syn, neurons[pre_id], neurons[post_id],
                            combined_mod, 1.0f, stdp_engine.lr_global);

                        float ca_plasticity = 0.0f;
                        for (auto& comp : dendrite_trees[post_id].compartments) {
                            ca_plasticity += comp.calc_calcium_gated_plasticity();
                        }
                        if (dendrite_trees[post_id].compartments.size() > 0)
                            ca_plasticity /= dendrite_trees[post_id].compartments.size();

                        float base_delta = syn.weight * 0.001f;
                        float ca_delta = stdp_engine.compute_calcium_modulated_update(
                            base_delta, ca_plasticity);
                        if (std::isfinite(ca_delta)) syn.weight += ca_delta;
                        if (syn.weight > 0.5f) syn.weight = 0.5f;
                        if (syn.weight < -0.5f) syn.weight = -0.5f;
                    }
                }
            }
        }
    }
}

float NeuronPopulation::compute_td_error(float reward, float gamma) const {
    float r = std::isfinite(reward) ? reward : 0.0f;
    float v_cur = std::isfinite(td_v_current) ? td_v_current : 0.0f;
    float v_prev = std::isfinite(td_v_previous) ? td_v_previous : 0.0f;
    float td = r + gamma * v_cur - v_prev;
    return std::isfinite(td) ? std::max(-1.0f, std::min(1.0f, td)) : 0.0f;
}
