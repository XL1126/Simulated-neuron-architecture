#include "cortical_brain.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

CorticalBrain::CorticalBrain(uint32_t total_neurons,
                              const std::vector<std::string>& concepts)
    : n_total(total_neurons), concept_list(concepts),
      step_counter(0), global_dopamine(0.5f),
      prediction_error(0.5f), prev_phi(0.0f),
      global_serotonin(0.5f), global_norepinephrine(0.3f),
      curiosity_drive(0.3f), social_drive(0.2f), exploration_drive(0.4f),
      surprise_level(0.5f), oscillatory_phase(0.0f),
      intrinsic_reward(0.0f), self_model_error(0.5f),
      filtered_self_model_error(0.5f),
      dopamine_baseline_ema(0.5f), ne_baseline_ema(0.3f),
      dopamine_baseline_sigma(0.1f), ne_baseline_sigma(0.1f),
      mi_max_eigenvalue(0.0f), last_mi_update(0),
      pca_mean(0.0f), pca_scale(1.0f),
      input_perturbed(false),
      cross_modal_learning_rate(0.005f),
      current_episode_start_step(0),
      world_prediction_error(0.5f), mi_penalty(0.0f), mi_penalty_weight(0.1f),
      global_activity_ema(0.1f), adaptive_noise_boost(0.0f),
      intrinsic_reward_state(0.0f),
      second_order_meta_confidence(0.5f),
      avg_perceptual_vividness(0.3f),
      self_prediction_error_floor(0.15f),
      prev_self_model_accuracy(0.5f),
      prev_global_ignition_cache(0.0f),
      previous_reward(0.0f),
      last_action_taken(0)
{
    uint32_t vis_n    = (uint32_t)(total_neurons * 0.15f);
    uint32_t mot_n    = (uint32_t)(total_neurons * 0.08333f);
    uint32_t hippo_n  = (uint32_t)(total_neurons * 0.18333f);
    uint32_t pfc_n    = (uint32_t)(total_neurons * 0.15f);
    uint32_t amyg_n   = (uint32_t)(total_neurons * 0.05f);
    uint32_t lang_n   = (uint32_t)(total_neurons * 0.08333f);
    uint32_t ws_n     = (uint32_t)(total_neurons * 0.20833f);
    uint32_t thal_n   = (uint32_t)(total_neurons * 0.04167f);
    uint32_t claus_n  = (uint32_t)(total_neurons * 0.025f);
    uint32_t dmn_n    = (uint32_t)(total_neurons * 0.025f);

    uint32_t sum = vis_n + mot_n + hippo_n + pfc_n + amyg_n + lang_n
                   + ws_n + thal_n + claus_n + dmn_n;
    uint32_t leftover = total_neurons - sum;
    ws_n += leftover;

    uint32_t cursor = 0;
    regions = {
        {"visual",      cursor, vis_n,   0, 0},
        {"motor",       cursor += vis_n,      mot_n,   8, 0},
        {"hippocampal", cursor += mot_n,      hippo_n, 0, 0},
        {"prefrontal",  cursor += hippo_n,    pfc_n,   0, 8},
        {"amygdala",    cursor += pfc_n,      amyg_n,  0, 0},
        {"language",    cursor += amyg_n,     lang_n,  0, (uint32_t)concepts.size()},
        {"workspace",   cursor += lang_n,     ws_n,    0, 0},
        {"thalamus",    cursor += ws_n,       thal_n,  0, 0},
        {"claustrum",   cursor += thal_n,     claus_n, 0, 0},
        {"dmn",         cursor += claus_n,    dmn_n,   0, 0},
    };

    for (auto& reg : regions) {
        region_index[reg.name] = region_index.size();
        populations.emplace_back(reg.n_neurons, 20);
    }

    visual_idx       = region_index["visual"];
    motor_idx        = region_index["motor"];
    hippocampal_idx  = region_index["hippocampal"];
    prefrontal_idx   = region_index["prefrontal"];
    amygdala_idx     = region_index["amygdala"];
    language_idx     = region_index["language"];
    workspace_idx    = region_index["workspace"];
    thalamus_idx     = region_index["thalamus"];
    claustrum_idx    = region_index["claustrum"];
    dmn_idx          = region_index["dmn"];

    concept_activity.resize(concepts.size(), 0.01f);
    self_state.resize(128, 0.01f);
    predicted_state.resize(128, 0.01f);
    self_prediction_weights.resize(128 * 128, 0.0f);
    drive_activations.resize(8, 0.3f);

    mi_matrix.resize(regions.size() * regions.size(), 0.0f);

    _setup_esn_prefrontal();
    _wire_all_regions();

    language_layer.init(concepts.size(), EMBED_DIM,
                        regions[language_idx].n_neurons);
    language_layer.initialize(concepts);

    std::vector<size_t> pc_units = {64, 32, EMBED_DIM};
    pc_network.init(PC_N_LEVELS, pc_units);

    auditory_buffer.resize(32, 0.0f);
    tactile_buffer.resize(16, 0.0f);
    vestibular_buffer.resize(8, 0.0f);
    place_cell_buffer.resize(16, 0.0f);

    size_t n_modalities = 5;
    cross_modal_weights.resize(n_modalities * n_modalities, 0.0f);
    for (size_t i = 0; i < n_modalities * n_modalities; i++) {
        cross_modal_weights[i] = ((float)(i * 2654435761) / 1e9f) * 0.05f;
    }

    current_episode_summary.resize(EMBED_DIM, 0.0f);

    self_perception.init();
    world_model.init();
    club_estimator.init();
    FastMath::init_fast_math(fast_exp_table);
    action_selector.init();
    meta_cog.init();
    intrinsic_value.init(128);
    state_word_mapper.init((int)concepts.size());
    narrative_self.init();
    lang_seq_gen.init();
    second_order_conf.clear();

    autobiographical_memory.init();
    counterfactual_engine.init();
    qualia_layer.init();
    intrinsic_motivation.init(128, 42);
    metacognition_monitor.init();
    spontaneous_thinker.init(43);
    semantic_grounding.init((int)concepts.size(), 32, 45);
    temporal_depth.init(47);
    theory_of_mind.init(49);
    goal_generator.init(51);
    emotion_system.init(53);
    active_inference.init(55);
    social_interaction.init(57);
    creative_generator.init(59);
    qualia_vividness_history.clear();
}

void CorticalBrain::_setup_esn_prefrontal() {
    auto& pfc_reg = regions[prefrontal_idx];
    uint32_t reservoir_n = std::min((uint32_t)ESN_RESERVOIR_SIZE, pfc_reg.n_neurons / 2);
    uint32_t readout_n = std::min((uint32_t)ESN_READOUT_SIZE,
        pfc_reg.n_neurons - reservoir_n);

    esn_reservoir_neurons.resize(reservoir_n);
    for (uint32_t i = 0; i < reservoir_n; i++) {
        esn_reservoir_neurons[i] = pfc_reg.base_id + i;
    }

    esn_readout_weights.resize(reservoir_n * readout_n, 0.0f);
    for (size_t i = 0; i < esn_readout_weights.size(); i++) {
        esn_readout_weights[i] = static_cast<float>(i * 1103515245 + 12345) / (float)0xFFFFFFFF * 0.01f;
    }
}

void CorticalBrain::_wire_all_regions() {
    for (auto& reg : regions) _wire_region(reg);
    _wire_region_interconnects();
    _add_recurrent_self_excitation();
}

void CorticalBrain::_wire_region(const BrainRegion& reg) {
    auto& pop = populations[region_index[reg.name]];
    std::vector<InbornSynapse> synapses;

    if (reg.name == "visual") {
        uint32_t v1_n = (uint32_t)(reg.n_neurons * 0.667f);
        uint32_t v2_n = reg.n_neurons - v1_n;
        synapses = circuit_builder.build_visual_hierarchical(
            reg.base_id, v1_n, reg.base_id + v1_n, v2_n, 8);
    }
    else if (reg.name == "motor") {
        synapses = circuit_builder.build_motor_somatotopic(
            reg.base_id, reg.n_neurons, 8, 4);
    }
    else if (reg.name == "hippocampal") {
        uint32_t dg_n = (uint32_t)(reg.n_neurons * 0.341f);
        uint32_t ca3_n = (uint32_t)(reg.n_neurons * 0.398f);
        uint32_t ca1_n = reg.n_neurons - dg_n - ca3_n;
        synapses = circuit_builder.build_hippocampal_structured(
            reg.base_id, dg_n,
            reg.base_id + dg_n, ca3_n,
            reg.base_id + dg_n + ca3_n, ca1_n,
            0.05f, 0.10f);
    }
    else if (reg.name == "prefrontal") {
        uint32_t reservoir_n = std::min((uint32_t)ESN_RESERVOIR_SIZE, reg.n_neurons / 2);
        uint32_t readout_n = std::min((uint32_t)ESN_READOUT_SIZE,
            reg.n_neurons - reservoir_n);
        synapses = circuit_builder.build_prefrontal_esn(
            reg.base_id, reg.n_neurons, reservoir_n, readout_n);
    }
    else if (reg.name == "amygdala") {
        synapses = circuit_builder.build_amygdala_valence(
            reg.base_id, reg.n_neurons);
    }
    else if (reg.name == "language") {
        synapses = circuit_builder.build_language_semantic(
            reg.base_id, reg.n_neurons, concept_list);
    }
    else if (reg.name == "workspace") {
        uint32_t n_sensory = (uint32_t)(reg.n_neurons * 0.25f);
        uint32_t n_competition = (uint32_t)(reg.n_neurons * 0.40f);
        uint32_t n_broadcast = reg.n_neurons - n_sensory - n_competition;
        synapses = circuit_builder.build_workspace_three_layer(
            reg.base_id, reg.n_neurons, n_sensory, n_competition, n_broadcast);
    }
    else if (reg.name == "thalamus") {
        synapses = circuit_builder.build_thalamic_relay(
            reg.base_id, reg.n_neurons);
    }
    else if (reg.name == "claustrum") {
        synapses = circuit_builder.build_claustrum_coordinator(
            reg.base_id, reg.n_neurons);
    }
    else if (reg.name == "dmn") {
        synapses = circuit_builder.build_default_mode_network(
            reg.base_id, reg.n_neurons);
    }

    for (auto& s : synapses) {
        uint32_t src = s.src % n_total;
        uint32_t dst = s.dst % n_total;
        pop.add_synapse(src, dst, s.weight, s.delay);
    }
}

void CorticalBrain::_add_recurrent_self_excitation() {
    for (size_t ri = 0; ri < regions.size(); ri++) {
        auto& reg = regions[ri];
        auto& pop = populations[ri];
        if (reg.name == "workspace") continue;
        size_t n = std::min((uint32_t)(reg.n_neurons * 0.3f), reg.n_neurons);
        float base_weight = 0.08f;
        uint32_t max_k = std::min(reg.n_neurons, (uint32_t)8);
        if (reg.name == "dmn") {
            n = std::min((uint32_t)(reg.n_neurons * 0.6f), reg.n_neurons);
            base_weight = 0.14f;
            max_k = std::min(reg.n_neurons, (uint32_t)16);
        }
        for (uint32_t i = 0; i < (uint32_t)n; i++) {
            uint32_t src = reg.base_id + i;
            float dist_threshold = (float)reg.n_neurons * 0.6f;
            for (uint32_t kj = 0; kj < max_k; kj++) {
                uint32_t dst = reg.base_id + ((i + kj * 31 + 17) % reg.n_neurons);
                if (src != dst) {
                    uint32_t dist = src > dst ? src - dst : dst - src;
                    float prob = (dist < dist_threshold) ? 0.9f : 0.1f;
                    float rval = (float)((src * 1103515245 + dst * 12345 + kj) % 10000) / 10000.0f;
                    if (rval < prob) {
                        pop.add_synapse(src, dst, base_weight, (uint8_t)(1 + (kj % 3)));
                    }
                }
            }
        }
    }
}

void CorticalBrain::_wire_region_interconnects() {
    float inter_prob = 0.08f;

    auto make_inter = [&](size_t src_i, size_t dst_i, float prob_mul,
                          float max_w, uint8_t max_d) {
        auto syns = circuit_builder.build_inter_region(
            regions[src_i].base_id, regions[src_i].n_neurons,
            regions[dst_i].base_id, regions[dst_i].n_neurons,
            inter_prob * prob_mul, max_w, max_d);
        for (auto& s : syns)
            populations[src_i].add_synapse(
                s.src % n_total, s.dst % n_total, s.weight, s.delay);
    };

    make_inter(visual_idx, motor_idx, 3.0f, 0.15f, 5);
    make_inter(motor_idx, visual_idx, 1.0f, 0.06f, 5);
    make_inter(visual_idx, hippocampal_idx, 3.0f, 0.12f, 3);
    make_inter(hippocampal_idx, visual_idx, 1.0f, 0.05f, 4);

    auto hippo_pfc = circuit_builder.build_inter_region(
        regions[hippocampal_idx].base_id, regions[hippocampal_idx].n_neurons,
        regions[prefrontal_idx].base_id, regions[prefrontal_idx].n_neurons,
        inter_prob * 3, 0.15f, 4);
    for (auto& s : hippo_pfc) {
        populations[hippocampal_idx].add_synapse(
            s.src % n_total, s.dst % n_total, s.weight, s.delay);
        populations[prefrontal_idx].add_synapse(
            s.dst % n_total, s.src % n_total, s.weight * 0.6f, s.delay);
    }

    auto pfc_lang = circuit_builder.build_inter_region(
        regions[prefrontal_idx].base_id, regions[prefrontal_idx].n_neurons,
        regions[language_idx].base_id, regions[language_idx].n_neurons,
        inter_prob * 3, 0.15f, 4);
    for (auto& s : pfc_lang) {
        populations[prefrontal_idx].add_synapse(
            s.src % n_total, s.dst % n_total, s.weight, s.delay);
        populations[language_idx].add_synapse(
            s.dst % n_total, s.src % n_total, s.weight * 0.5f, s.delay);
    }

    make_inter(language_idx, motor_idx, 2.0f, 0.08f, 6);
    make_inter(motor_idx, language_idx, 1.0f, 0.04f, 6);
    make_inter(amygdala_idx, prefrontal_idx, 4.0f, 0.25f, 3);
    make_inter(amygdala_idx, motor_idx, 3.0f, 0.2f, 3);
    make_inter(hippocampal_idx, amygdala_idx, 2.0f, 0.12f, 3);
    make_inter(visual_idx, amygdala_idx, 2.0f, 0.15f, 2);

    make_inter(visual_idx, thalamus_idx, 3.0f, 0.15f, 2);
    make_inter(motor_idx, thalamus_idx, 2.0f, 0.1f, 2);
    make_inter(hippocampal_idx, thalamus_idx, 2.0f, 0.12f, 3);
    make_inter(amygdala_idx, thalamus_idx, 2.0f, 0.15f, 2);
    make_inter(language_idx, thalamus_idx, 2.0f, 0.1f, 3);
    make_inter(thalamus_idx, prefrontal_idx, 3.0f, 0.15f, 2);
    make_inter(thalamus_idx, visual_idx, 2.0f, 0.08f, 3);
    make_inter(thalamus_idx, motor_idx, 2.0f, 0.08f, 3);

    for (size_t ri = 0; ri < regions.size(); ri++) {
        if (ri == claustrum_idx) continue;
        make_inter(ri, claustrum_idx, 1.5f, 0.1f, 3);
        make_inter(claustrum_idx, ri, 1.5f, 0.08f, 3);
    }

    make_inter(prefrontal_idx, dmn_idx, 3.0f, 0.15f, 3);
    make_inter(hippocampal_idx, dmn_idx, 3.0f, 0.12f, 3);
    make_inter(dmn_idx, prefrontal_idx, 3.0f, 0.12f, 3);
    make_inter(dmn_idx, hippocampal_idx, 2.0f, 0.08f, 4);
    make_inter(amygdala_idx, dmn_idx, 1.5f, 0.1f, 3);
    make_inter(claustrum_idx, dmn_idx, 1.5f, 0.08f, 3);

    for (size_t ri = 0; ri < regions.size(); ri++) {
        if (ri == workspace_idx) continue;
        make_inter(ri, workspace_idx, 1.5f, 0.2f, 3);
        make_inter(workspace_idx, ri, 1.5f, 0.15f, 2);
    }

    uint32_t ca3_start = regions[hippocampal_idx].base_id
        + (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    uint32_t ca3_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    swr_engine.init(ca3_n);
    replay_engine.init(ca3_n);

    uint32_t ca1_start = regions[hippocampal_idx].base_id
        + (uint32_t)(regions[hippocampal_idx].n_neurons * 0.5f);
    uint32_t ca1_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.159f);
    cortical_rx.init(ca1_n, regions[prefrontal_idx].n_neurons);
}

void CorticalBrain::step(float world_reward) {
    step_counter++;

    float surprise = std::abs(world_reward - previous_reward);
    previous_reward = world_reward;
    surprise_level = surprise_level * 0.95f + surprise * 0.05f;

    if (surprise > 0.3f) input_perturbed = true;

    global_dopamine = std::max(0.0f, std::min(1.0f,
        global_dopamine + world_reward * 0.3f + 0.0005f));
    global_norepinephrine = std::max(0.0f, std::min(1.0f,
        global_norepinephrine * 0.98f + surprise * 0.3f));
    global_serotonin = std::max(0.0f, std::min(1.0f,
        global_serotonin * 0.99f + world_reward * 0.1f + 0.5f));

    _update_neuromodulator_baselines();

    curiosity_drive = std::max(0.0f, std::min(1.0f,
        curiosity_drive * 0.95f + global_norepinephrine * 0.1f));
    exploration_drive = std::max(0.0f, std::min(1.0f,
        exploration_drive * 0.93f + surprise_level * 0.1f));

    oscillatory_phase += 0.0628f;
    if (oscillatory_phase > 6.283f) oscillatory_phase -= 6.283f;

    float effective_dopamine = global_dopamine
        + 0.1f * std::sin(oscillatory_phase) * global_norepinephrine;
    effective_dopamine = std::max(0.0f, std::min(1.0f, effective_dopamine));

    for (size_t i = 0; i < populations.size(); i++) {
        float region_noise = 0.025f + 0.005f * (float)(step_counter % 7) + adaptive_noise_boost;
        if (i == workspace_idx)
            region_noise += global_norepinephrine * 0.015f;
        if (i == hippocampal_idx)
            region_noise += curiosity_drive * 0.008f;
        if (i == dmn_idx)
            region_noise += exploration_drive * 0.005f;
        populations[i].update(step_counter, region_noise, effective_dopamine);
    }

    float total_fires = 0;
    for (auto& p : populations) {
        total_fires += (float)p.get_current_fires().size();
    }
    float act = total_fires / (float)n_total;
    global_activity_ema = global_activity_ema * 0.95f + act * 0.05f;

    if (global_activity_ema < 0.06f) {
        adaptive_noise_boost = std::min(0.08f, adaptive_noise_boost + 0.001f);
    } else if (global_activity_ema > 0.20f) {
        adaptive_noise_boost = std::max(0.0f, adaptive_noise_boost - 0.002f);
    } else {
        adaptive_noise_boost *= 0.998f;
    }

    _apply_amygdala_modulation();
    _apply_global_workspace();
    _apply_oscillatory_coupling();
    _collect_self_state();
    _update_self_model();
    _update_world_model();
    _update_mi_penalty();
    _update_meta_cognition();

    for (size_t ri = 0; ri < regions.size(); ri++) {
        auto& fires = populations[ri].get_current_fires();
        std::vector<float> acts(regions[ri].n_neurons > 0 ?
            std::min((size_t)regions[ri].n_neurons, (size_t)1024) : 64, 0.0f);
        size_t n = std::min(acts.size(), fires.size());
        for (size_t i = 0; i < n; i++) {
            acts[i] = fires[i].strength;
        }
        intrinsic_motivation.update_region_prediction((int)ri, acts, acts.size());
    }

    std::vector<float> thought_vec = read_thought_vector();
    qualia_layer.forward(
        thought_vec,
        auditory_buffer, tactile_buffer, vestibular_buffer,
        self_state, global_dopamine, global_norepinephrine, 1.0f);
    qualia_vividness_history.push_back(qualia_layer.get_vividness());
    if (qualia_vividness_history.size() > 50) qualia_vividness_history.pop_front();
    avg_perceptual_vividness = 0.0f;
    for (auto v : qualia_vividness_history) avg_perceptual_vividness += v;
    if (!qualia_vividness_history.empty())
        avg_perceptual_vividness /= (float)qualia_vividness_history.size();

    narrative_self.update(self_state,
        episodic_summaries.empty() ? std::vector<float>(64, 0.0f) : episodic_summaries.back(),
        meta_cog.get_confidence(), meta_cog.get_surprise(),
        meta_cog.get_valence(), intrinsic_reward_state);

    second_order_conf.push_back(meta_cog.get_confidence());
    if (second_order_conf.size() > 30) second_order_conf.pop_front();
    if (second_order_conf.size() >= 5) {
        float mean = 0, varsum = 0;
        for (auto v : second_order_conf) mean += v;
        mean /= (float)second_order_conf.size();
        for (auto v : second_order_conf) varsum += (v - mean) * (v - mean);
        float var = varsum / (float)second_order_conf.size();
        second_order_meta_confidence = 1.0f / (1.0f + var * 10.0f);
    }

    std::vector<float> world_state(64, 0.0f);
    {
        auto& vis_fires = populations[visual_idx].get_current_fires();
        for (auto& f : vis_fires) {
            uint32_t idx = f.neuron_id % 64;
            world_state[idx] += f.strength * 0.05f;
        }
    }

    float sig = autobiographical_memory.compute_significance(
        world_state, self_state, previous_reward, meta_cog.get_surprise());
    autobiographical_memory.store(world_state, self_state,
        last_action_taken, previous_reward, sig);
    counterfactual_engine.record_experience(world_state,
        last_action_taken, previous_reward);

    auto world_pred = world_model.predict_next(world_state);
    auto cf_alts = counterfactual_engine.simulate_alternatives(
        last_action_state, last_action_taken, previous_reward,
        world_state, world_pred);
    counterfactual_engine.learn_from_regret(cf_alts, world_state);

    if (step_counter % 500 == 499) {
        autobiographical_memory.consolidate();
    }

    _compute_intrinsic_motivation();
    _update_esn_readout();
    _process_predictive_coding();
    _update_cross_modal_associations();
    qualia_layer.learn(previous_reward);

    float sensory_mag = 0.0f;
    {
        auto& vis_fires = populations[visual_idx].get_current_fires();
        for (auto& f : vis_fires) sensory_mag += f.strength;
        auto& aud_fires = populations[thalamus_idx].get_current_fires();
        for (auto& f : aud_fires) sensory_mag += f.strength * 0.3f;
    }
    sensory_mag = std::min(1.0f, sensory_mag * 0.02f);

    std::vector<float> episodic_ctx;
    if (!episodic_summaries.empty()) {
        episodic_ctx = episodic_summaries.back();
    }

    spontaneous_thinker.think(
        self_state,
        episodic_ctx,
        sensory_mag,
        global_dopamine,
        global_norepinephrine,
        meta_cog.get_surprise());

    if (spontaneous_thinker.should_trigger_memory_recall()) {
        auto probe = spontaneous_thinker.get_memory_probe();
        _recall_episodic_memory(probe);
    }

    std::vector<float> sensory_features;
    {
        auto& vis_fires = populations[visual_idx].get_current_fires();
        sensory_features.resize(32, 0.0f);
        for (auto& f : vis_fires) {
            uint32_t idx = f.neuron_id % 32;
            if (idx < 32) sensory_features[idx] += f.strength * 0.05f;
        }
    }

    semantic_grounding.assoc_learn(
        sensory_features,
        concept_activity,
        previous_reward,
        self_model_error);

    auto grounded = semantic_grounding.get_grounded_concepts(
        sensory_features, concept_activity, 0.4f);

    auto dmn_modulation = spontaneous_thinker.get_concept_modulation(
        (int)concept_activity.size());

    _update_drives();
    _update_concept_activities();

    for (size_t ci = 0; ci < concept_activity.size(); ci++) {
        if (ci < grounded.size()) {
            concept_activity[ci] = concept_activity[ci] * 0.35f + grounded[ci] * 0.65f;
        }
        if (ci < dmn_modulation.size()) {
            concept_activity[ci] = std::min(1.0f,
                concept_activity[ci] + dmn_modulation[ci] * 0.20f);
        }
    }

    auto dmn_ctx = spontaneous_thinker.get_language_context();
    language_layer.inject_dmn_context(dmn_ctx, spontaneous_thinker.spontaneity);

    for (size_t ci = 0; ci < concept_activity.size(); ci++) {
        float dmn_val = 0.0f;
        if (ci < dmn_ctx.size()) dmn_val = dmn_ctx[ci % dmn_ctx.size()];
        size_t hash_ci = ci * 7 + step_counter;
        dmn_val += (float)((hash_ci * 1103515245) % 10000) / 10000.0f * 0.02f;
        concept_activity[ci] = std::min(1.0f, std::max(0.0f,
            concept_activity[ci] + dmn_val * spontaneous_thinker.spontaneity * 0.60f));
    }

    if (step_counter % 250 == 249 && spontaneous_thinker.spontaneity > 0.7f) {
        uint32_t seed_off = (uint32_t)(step_counter * 1103515245);
        for (size_t ci = 0; ci < concept_activity.size() / 3; ci++) {
            size_t idx = (ci * 7 + seed_off) % concept_activity.size();
            float rv = (float)((seed_off + ci * 12345) % 10000) / 10000.0f;
            concept_activity[idx] = concept_activity[idx] * 0.4f + rv * 0.6f;
        }
    }

    if (step_counter % 100 == 99) {
        std::vector<float> episode_sig = current_episode_summary;
        float sig = meta_cog.get_surprise() * 0.4f + std::abs(previous_reward) * 0.4f + 0.2f;
        spontaneous_thinker.store_pattern(episode_sig, sig, (int)step_counter);
    }

    float significance = meta_cog.get_surprise() * 0.5f + std::abs(previous_reward) * 0.5f;
    std::vector<float> episodic_context;
    for (size_t i = 0; i < 32 && i < concept_activity.size(); i++)
        episodic_context.push_back(concept_activity[i]);
    temporal_depth.record_slice(self_state, episodic_context, significance,
        previous_reward > 0.0f ? 1.0f : (previous_reward < 0.0f ? -1.0f : 0.0f));

    auto episodic_summary = autobiographical_memory.get_consolidated_self();
    float world_change = 0.0f;
    {
        float prev = 0.0f, cur = 0.0f;
        for (size_t i = 0; i < concept_activity.size(); i++) {
            cur += std::abs(concept_activity[i]);
        }
        for (auto& v : episodic_context) prev += std::abs(v);
        world_change = std::abs(cur - prev) / std::max(1.0f, (float)concept_activity.size());
    }
    temporal_depth.update_temporal_chain(self_state, episodic_summary,
        meta_cog.get_confidence(), world_change);

    std::vector<float> trend_vector(64, 0.0f);
    {
        auto& pfc_fires = populations[prefrontal_idx].get_current_fires();
        std::vector<float> pfc_act(64, 0.0f);
        for (auto& f : pfc_fires) {
            uint32_t idx = f.neuron_id % 64;
            if (idx < 64) pfc_act[idx] += f.strength * 0.03f;
        }
        for (int d = 0; d < 64; d++) trend_vector[d] = pfc_act[d];
    }
    temporal_depth.project_future(self_state, episodic_context, trend_vector,
        global_dopamine);

    if (step_counter % 200 == 199) {
        float self_change = 0.0f;
        for (size_t i = 0; i < self_state.size(); i++) {
            self_change += std::abs(self_state[i] - predicted_state[i]);
        }
        self_change /= (float)self_state.size();
        temporal_depth.update_chapters(self_state, self_change, 0.3f);
    }

    if (step_counter % 20 == 19) {
        temporal_depth.evaluate_anticipation(self_state, 1.0f);
    }

    std::vector<float> other_obs(64, 0.0f);
    for (int d = 0; d < 64; d++) {
        other_obs[d] = self_state[d] * 0.7f + episodic_context[d % episodic_context.size()] * 0.3f;
        float noise = (float)((d * 1103515245 + step_counter * 25214903917ULL) % 10000) / 10000.0f * 0.1f;
        other_obs[d] = std::max(0.0f, std::min(1.0f, other_obs[d] + noise));
    }

    if (step_counter % 50 == 49) {
        theory_of_mind.perceive_agent(0, other_obs, episodic_context, previous_reward);
        theory_of_mind.update_beliefs(0, episodic_context, self_state);
        theory_of_mind.infer_intentions(0, trend_vector);
    }
    theory_of_mind.update_self_other_boundary(self_state, other_obs);
    theory_of_mind.update_social_context(episodic_context,
        previous_reward > 0.0f ? 0.6f : 0.3f, previous_reward);

    if (step_counter % 200 == 0) {
        float curiosity = curiosity_drive * 1.5f;
        goal_generator.generate_candidates(
            self_state, episodic_context, concept_activity,
            temporal_depth.get_temporal_context(),
            curiosity, 1.0f - meta_cog.get_surprise(),
            narrative_self.get_stability(), curiosity);
    }

    goal_generator.evaluate_goals(self_state, previous_reward, world_change);
    goal_generator.select_active_goal(self_state, global_dopamine,
        meta_cog.get_confidence());
    goal_generator.update_progress(self_state, episodic_context);

    goal_generator.decay_stale_goals(5000);

    auto goal_dir = goal_generator.get_goal_direction();
    for (size_t ci = 0; ci < concept_activity.size() && ci < goal_dir.size(); ci++) {
        concept_activity[ci] = std::min(1.0f, std::max(0.0f,
            concept_activity[ci] + goal_dir[ci] * 0.05f));
    }

    float temporal_contrast = 0.0f;
    {
        std::vector<float> prev_self = temporal_depth.get_temporal_context();
        for (size_t i = 0; i < std::min(self_state.size(), prev_self.size()); i++) {
            temporal_contrast += std::abs(self_state[i] - prev_self[i]);
        }
        size_t n = std::min(self_state.size(), prev_self.size());
        if (n > 0) temporal_contrast /= (float)n;
    }

    emotion_system.update(
        previous_reward,
        curiosity_drive,
        prediction_error,
        theory_of_mind.social_awareness * 0.5f,
        filtered_self_model_error,
        temporal_contrast,
        concept_activity);

    emotion_system.regulate_emotions(global_serotonin, global_norepinephrine);

    if (step_counter % 50 == 49 && temporal_contrast > 0.3f) {
        emotion_system.store_emotional_memory(self_state, temporal_contrast);
    }

    auto emotion_vec = emotion_system.get_emotion_vector();

    std::vector<float> world_pred_vec = world_model.get_last_prediction();
    if (step_counter % 100 == 99) {
        active_inference.generate_plans(
            self_state,
            goal_dir.empty() ? std::vector<float>(32, 0.5f) : goal_dir,
            world_pred_vec.empty() ? std::vector<float>(32, 0.3f) : world_pred_vec,
            emotion_vec,
            global_dopamine,
            curiosity_drive);
    }

    active_inference.evaluate_plans(
        self_state,
        goal_dir.empty() ? self_state : goal_dir,
        1.0f - global_dopamine,
        0.95f);

    active_inference.select_plan(
        meta_cog.get_confidence(),
        exploration_drive);

    auto planned_action = active_inference.get_planned_action();
    for (size_t ci = 0; ci < concept_activity.size() && ci < planned_action.size(); ci++) {
        concept_activity[ci] = std::min(1.0f, std::max(0.0f,
            concept_activity[ci] + planned_action[ci] * 0.04f));
    }

    std::vector<float> social_obs(16, 0.0f);
    for (int d = 0; d < 16; d++) {
        social_obs[d] = self_state[d] * 0.6f
            + (d < (int)concept_activity.size() ? concept_activity[d] : 0.0f) * 0.4f;
    }
    social_interaction.add_or_update_agent(1, social_obs,
        episodic_context.empty() ? std::vector<float>(8, 0.0f) : std::vector<float>(episodic_context.begin(), episodic_context.begin() + std::min<size_t>(8, episodic_context.size())),
        {emotion_system.current_emotion.joy, emotion_system.current_emotion.trust,
         emotion_system.current_emotion.anticipation, emotion_system.current_emotion.sadness});

    if (step_counter % 150 == 149) {
        auto interaction = social_interaction.simulate_interaction(
            1, planned_action, emotion_vec,
            active_inference.action_confidence);
        social_interaction.learn_from_interaction(interaction, previous_reward);
    }

    social_interaction.update_relationships(0.01f);

    if (step_counter % 200 == 0) {
        creative_generator.generate_ideas(
            concept_activity, self_state, emotion_vec,
            episodic_context, temporal_depth.get_temporal_context(),
            spontaneous_thinker.spontaneity, global_dopamine);

        creative_generator.evaluate_ideas(
            self_state, episodic_context,
            meta_cog.get_confidence());

        creative_generator.select_best_idea(exploration_drive);
    }

    auto creative_mod = creative_generator.get_creative_modulation(
        concept_activity, (int)concept_activity.size());
    for (size_t ci = 0; ci < concept_activity.size() && ci < creative_mod.size(); ci++) {
        concept_activity[ci] = std::min(1.0f, std::max(0.0f,
            concept_activity[ci] + creative_mod[ci] * 0.15f));
    }

    auto novel_combo = creative_generator.get_novel_concept_combination(
        (int)concept_activity.size());
    if (step_counter % 400 == 399) {
        for (size_t ci = 0; ci < concept_activity.size() && ci < novel_combo.size(); ci++) {
            concept_activity[ci] = concept_activity[ci] * 0.7f + novel_combo[ci] * 0.3f;
        }
    }

    _store_episodic_memory();
    _update_consciousness();

    _process_brain_state();
    _step_swr_replay();
}

void CorticalBrain::_update_neuromodulator_baselines() {
    float ema_alpha = 0.001f;
    dopamine_baseline_ema = dopamine_baseline_ema * (1.0f - ema_alpha)
                            + global_dopamine * ema_alpha;
    ne_baseline_ema = ne_baseline_ema * (1.0f - ema_alpha)
                      + global_norepinephrine * ema_alpha;

    float sigma_alpha = 0.0005f;
    float d_diff = global_dopamine - dopamine_baseline_ema;
    float ne_diff = global_norepinephrine - ne_baseline_ema;
    dopamine_baseline_sigma = std::max(0.01f,
        dopamine_baseline_sigma * (1.0f - sigma_alpha)
        + std::abs(d_diff) * sigma_alpha);
    ne_baseline_sigma = std::max(0.01f,
        ne_baseline_sigma * (1.0f - sigma_alpha)
        + std::abs(ne_diff) * sigma_alpha);
}

float CorticalBrain::_sliding_window_normalize(float value,
    std::deque<float>& history, size_t max_size) {
    history.push_back(value);
    if (history.size() > max_size) history.pop_front();

    if (history.size() < 5) return std::max(0.0f, std::min(1.0f, value));

    float mean = 0.0f;
    for (auto v : history) mean += v;
    mean /= (float)history.size();

    float var = 0.0f;
    for (auto v : history) var += (v - mean) * (v - mean);
    var /= (float)history.size();
    float sigma = std::sqrt(std::max(1e-8f, var));

    float robust_sigma = sigma < 1e-6f ? 0.02f : sigma;

    float z_score = (value - mean) / (robust_sigma + 1e-8f);

    float sigmoid_norm = 1.0f / (1.0f + std::exp(-z_score * 1.8f));

    float absolute_scale = std::min(1.0f, value * 4.0f);
    float normalized = 0.4f * sigmoid_norm + 0.6f * absolute_scale;

    return std::max(0.0f, std::min(1.0f, normalized));
}

void CorticalBrain::_apply_oscillatory_coupling() {
    float theta_phase = oscillatory_phase;
    float alpha_phase = oscillatory_phase * 1.7f;
    if (alpha_phase > 6.283f) alpha_phase -= 6.283f;

    float thalamic_rhythm = (std::sin(theta_phase) + 1.0f) * 0.5f;

    auto& ws_pop = populations[workspace_idx];
    auto& ws_fires = ws_pop.get_current_fires();
    float ws_active = ws_fires.empty() ? 0.0f : std::min(1.0f, ws_fires.size() / 15.0f);

    for (size_t ri = 0; ri < populations.size(); ri++) {
        if (ri == workspace_idx) continue;
        float coupling = thalamic_rhythm * 0.03f * ws_active
                         + std::sin(alpha_phase + (float)ri * 0.9f) * 0.015f;
        float new_dop = std::max(0.0f, std::min(1.0f,
            (float)populations[ri].get_dopamine() + coupling));
        populations[ri].set_dopamine(new_dop);
    }
}

void CorticalBrain::_update_drives() {
    drive_activations[0] = curiosity_drive;
    drive_activations[1] = exploration_drive;
    drive_activations[2] = social_drive;
    drive_activations[3] = global_dopamine;
    drive_activations[4] = 0.3f + global_norepinephrine * 0.4f;
    drive_activations[5] = surprise_level;
    drive_activations[6] = prediction_error;
    drive_activations[7] = global_serotonin;
}

void CorticalBrain::inject_sensory(const std::vector<float>& features) {
    auto& vis_pop = populations[visual_idx];
    uint32_t v1_n = (uint32_t)(regions[visual_idx].n_neurons * 0.667f);
    size_t nf = std::min(features.size(), (size_t)32);

    for (size_t i = 0; i < nf; i++) {
        if (std::abs(features[i]) > 0.01f) {
            uint32_t nid = regions[visual_idx].base_id
                + ((uint32_t)(i * 7 + (uint32_t)(std::abs(features[i]) * 100))) % v1_n;
            vis_pop.inject_spike(nid, std::abs(features[i]) * 5.0f, 1 + (uint32_t)(i % 4));
        }
    }

    auto& hippo_pop = populations[hippocampal_idx];
    uint32_t dg_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    for (size_t i = 0; i < std::min(nf, (size_t)8); i++) {
        uint32_t nid = regions[hippocampal_idx].base_id + (uint32_t)i % dg_n;
        hippo_pop.inject_spike(nid, std::abs(features[i]) * 1.5f, 2 + (uint32_t)(i % 3));
    }

    auto& thal_pop = populations[thalamus_idx];
    for (size_t i = 0; i < std::min(nf, (size_t)6); i++) {
        uint32_t nid = regions[thalamus_idx].base_id
            + (uint32_t)(i * 11) % regions[thalamus_idx].n_neurons;
        thal_pop.inject_spike(nid, std::abs(features[i]) * 0.8f, 1 + (uint32_t)(i % 3));
    }
}

void CorticalBrain::inject_text(const std::string& text) {
    input_perturbed = true;

    auto& lang_pop = populations[language_idx];
    uint32_t n_per = regions[language_idx].n_neurons /
                     std::max((uint32_t)concept_list.size(), (uint32_t)1);
    if (n_per < 8) n_per = 8;

    std::istringstream iss(text);
    std::string word;
    size_t wi = 0;
    while (iss >> word) {
        uint32_t best_ci = 0;
        size_t best_match = 0;
        for (size_t ci = 0; ci < concept_list.size(); ci++) {
            size_t match = 0;
            size_t pos = 0;
            while ((pos = concept_list[ci].find(word[0], pos)) != std::string::npos) {
                match++; pos++;
                if (match > best_match) {
                    best_match = match;
                    best_ci = (uint32_t)ci;
                }
            }
        }
        if (best_match == 0) {
            std::hash<std::string> hasher;
            uint32_t h32 = (uint32_t)(hasher(word) % (uint32_t)concept_list.size());
            best_ci = h32;
        }

        uint32_t start = regions[language_idx].base_id + best_ci * n_per;
        for (uint32_t k = 0; k < std::min(n_per / 2, (uint32_t)5); k++) {
            uint32_t nid = start + k;
            if (nid < regions[language_idx].base_id + regions[language_idx].n_neurons) {
                lang_pop.inject_spike(nid, 0.6f + 0.1f * (float)wi, 1 + (uint32_t)(wi % 5));
            }
        }
        wi++;
    }

    auto& pfc_pop = populations[prefrontal_idx];
    for (size_t i = 0; i < std::min(wi * 2, (size_t)12); i++) {
        uint32_t nid = regions[prefrontal_idx].base_id + (uint32_t)i;
        pfc_pop.inject_spike(nid, 0.3f, 2 + (uint32_t)(i % 4));
    }

    auto& thal_pop = populations[thalamus_idx];
    for (size_t i = 0; i < std::min(wi, (size_t)4); i++) {
        uint32_t nid = regions[thalamus_idx].base_id
            + (uint32_t)(i * 17) % regions[thalamus_idx].n_neurons;
        thal_pop.inject_spike(nid, 0.2f, 1 + (uint32_t)(i % 3));
    }
}

void CorticalBrain::inject_reward(float reward) {
    auto& amyg_pop = populations[amygdala_idx];
    uint32_t mid = regions[amygdala_idx].n_neurons / 2;
    uint32_t base = regions[amygdala_idx].base_id;
    uint32_t start = (reward > 0) ? base : base + mid;
    uint32_t end   = (reward > 0) ? base + mid : base + regions[amygdala_idx].n_neurons;
    uint32_t count = std::min((uint32_t)15, (end - start) / 3);

    for (uint32_t k = 0; k < count; k++) {
        amyg_pop.inject_spike(start + k, std::abs(reward) * 3.0f, 1 + (k % 3));
    }
}

std::vector<uint32_t> CorticalBrain::read_motor_output() const {
    auto& mot_pop = populations[motor_idx];
    auto& fires = mot_pop.get_current_fires();
    std::vector<uint32_t> result;
    for (auto& f : fires) {
        if (result.size() >= 10) break;
        result.push_back(f.neuron_id);
    }
    return result;
}

std::vector<float> CorticalBrain::read_thought_vector() const {
    std::vector<float> thought(256, 0.0f);
    std::vector<size_t> source_regions = {
        visual_idx, hippocampal_idx, prefrontal_idx, language_idx,
        workspace_idx, thalamus_idx, claustrum_idx, dmn_idx
    };
    for (size_t ri : source_regions) {
        auto& fires = populations[ri].get_current_fires();
        for (auto& f : fires) {
            uint32_t idx = f.neuron_id % 256;
            thought[idx] += f.strength * 0.08f;
        }
    }
    thought[0] += curiosity_drive * 0.5f;
    thought[1] += exploration_drive * 0.5f;
    thought[2] += global_dopamine * 0.5f;
    thought[3] += prediction_error * 0.5f;
    thought[4] += filtered_self_model_error * 0.5f;
    thought[5] += intrinsic_reward * 0.5f;
    for (size_t i = 0; i < std::min(self_state.size(), (size_t)32); i++) {
        thought[10 + i] += self_state[i] * 0.3f;
    }

    float norm = 0.0f;
    for (auto v : thought) norm += v * v;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& v : thought) v /= norm;
    }
    return thought;
}

std::string CorticalBrain::read_output_text() const {
    auto& lang_pop = populations[language_idx];
    auto& lang_fires = lang_pop.get_current_fires();

    std::vector<float> lang_activity(regions[language_idx].n_neurons, 0.0f);
    for (auto& f : lang_fires) {
        uint32_t local_id = f.neuron_id - regions[language_idx].base_id;
        if (local_id < lang_activity.size()) {
            lang_activity[local_id] = std::min(1.0f,
                lang_activity[local_id] + f.strength * 0.1f);
        }
    }

    auto& pfc_fires = populations[prefrontal_idx].get_current_fires();
    for (auto& f : pfc_fires) {
        uint32_t idx = f.neuron_id % lang_activity.size();
        lang_activity[idx] = std::min(1.0f,
            lang_activity[idx] + f.strength * 0.04f);
    }

    auto& dmn_fires = populations[dmn_idx].get_current_fires();
    for (auto& f : dmn_fires) {
        uint32_t idx = f.neuron_id % lang_activity.size();
        lang_activity[idx] = std::min(1.0f,
            lang_activity[idx] + f.strength * 0.02f);
    }

    auto thought = read_thought_vector();

    auto memories = autobiographical_memory.recall(
        narrative_self.who_am_i(), thought, 3, 0.1f);

    float emotional_tone = meta_cog.get_valence();
    if (!memories.empty()) {
        for (auto& mem : memories) {
            emotional_tone += mem.outcome * 0.15f;
        }
    }
    emotional_tone = std::max(-1.0f, std::min(1.0f, emotional_tone));

    auto seq = lang_seq_gen.generate_sequence(
        thought, narrative_self.who_am_i(),
        emotional_tone, meta_cog.get_arousal(),
        meta_cog.get_confidence(),
        language_layer.get_embedding_matrix(),
        (int)concept_list.size(), step_counter);

    if (!seq.empty()) {
        std::string combined;
        for (size_t si = 0; si < seq.size(); si++) {
            if (si > 0) combined += " ";
            if (seq[si].first >= 0 && seq[si].first < (int)concept_list.size())
                combined += concept_list[seq[si].first];
        }
        if (!combined.empty()) {
            float happiness = std::max(0.0f, emotional_tone);
            if (happiness > 0.5f && combined.find("good") == std::string::npos) {
                if (combined.length() < 20) combined += " good";
            }
            if (emotional_tone < -0.4f && combined.find("bad") == std::string::npos) {
                if (combined.length() < 20) combined += " bad";
            }
            return combined;
        }
    }

    int mapped_idx = state_word_mapper.map_to_word(
        thought,
        meta_cog.get_valence(), meta_cog.get_arousal(),
        language_layer.get_embedding_matrix(),
        (int)concept_list.size(), step_counter);

    if (mapped_idx >= 0 && mapped_idx < (int)concept_list.size()) {
        return concept_list[mapped_idx];
    }

    std::string output = const_cast<LanguageLayer&>(language_layer).generate_output(
        global_dopamine, global_norepinephrine,
        filtered_self_model_error, curiosity_drive,
        concept_activity, lang_activity, step_counter);

    const_cast<CorticalBrain*>(this)->_recall_episodic_memory(
        const_cast<LanguageLayer&>(language_layer).project_neural_activity(lang_activity));

    return output;
}

ConsciousnessState CorticalBrain::read_consciousness() const {
    ConsciousnessState cs;
    cs.phi = 0.0f;
    cs.phi_base = 0.0f;
    cs.phi_surprise = 0.0f;
    cs.global_ignition = 0.0f;
    cs.self_prediction_error = filtered_self_model_error;
    cs.coherence = 0.0f;
    cs.attention_focus = global_dopamine;

    for (auto& p : populations) {
        float act = (float)p.get_current_fires().size() / std::max(1.0f, (float)p.size());
        cs.region_activities.push_back(act);
    }

    auto& ws_fires = populations[workspace_idx].get_current_fires();
    std::set<uint32_t> seen;
    for (auto& f : ws_fires) {
        if (cs.active_concepts.size() < 10) {
            uint32_t ci = f.neuron_id % concept_list.size();
            if (seen.find(ci) == seen.end()) {
                seen.insert(ci);
                cs.active_concepts.push_back(concept_list[ci]);
            }
        }
    }

    if (!phi_history.empty()) {
        cs.phi = phi_history.back();
        auto& phi_copy = phi_history;
        if (phi_copy.size() >= 5) {
            size_t idx = phi_copy.size() - 5;
            float sum = 0.0f;
            for (size_t i = idx; i < phi_copy.size(); i++) sum += phi_copy[i];
            cs.phi_base = sum / (float)(phi_copy.size() - idx);
        }
        float recent = phi_copy.back();
        cs.phi_surprise = std::max(0.0f, recent - cs.phi_base);
    }
    {
        auto& ws_pop = populations[workspace_idx];
        auto& all_ws_fires = ws_pop.get_current_fires();
        float ws_size = (float)ws_pop.size();

        float n_fires = (float)all_ws_fires.size();
        float firing_fraction = ws_size > 0.0f ? n_fires / ws_size : 0.0f;

        float total_ws_act = 0.0f, max_ws_act = 0.0f;
        float top5_total = 0.0f;
        std::vector<float> strengths;
        strengths.reserve(all_ws_fires.size());
        for (auto& f : all_ws_fires) {
            total_ws_act += f.strength;
            if (f.strength > max_ws_act) max_ws_act = f.strength;
            strengths.push_back(f.strength);
        }
        std::sort(strengths.begin(), strengths.end(), std::greater<float>());
        for (size_t i = 0; i < std::min(strengths.size(), (size_t)5); i++)
            top5_total += strengths[i];

        float concentration = (total_ws_act > 0.001f) ? top5_total / total_ws_act : 0.0f;

        float participation = std::min(1.0f, firing_fraction * 5.0f);

        float inverted_u = 1.0f - std::abs(participation - 0.30f) * 3.0f;
        inverted_u = std::max(0.0f, std::min(1.0f, inverted_u));

        float ws_entropy = 0.0f;
        if (strengths.size() > 1) {
            for (auto s : strengths) {
                float p = s / (total_ws_act + 1e-8f);
                if (p > 1e-8f) ws_entropy -= p * std::log(p + 1e-8f);
            }
            float max_ent = std::log((float)strengths.size());
            if (max_ent > 1e-8f) ws_entropy /= max_ent;
        }

        float entropy_factor = 0.2f + 0.8f * ws_entropy;

        float raw_ignition = inverted_u * (0.35f + 0.65f * concentration) * entropy_factor;

        cs.global_ignition = raw_ignition * 0.80f
                             + prev_global_ignition_cache * 0.20f;
        prev_global_ignition_cache = cs.global_ignition;
        cs.global_ignition = std::min(1.0f, cs.global_ignition);
    }

    cs.coherence = 0.0f;
    if (cs.region_activities.size() >= 3) {
        float var = 0.0f;
        float mean_act = 0.0f;
        for (auto a : cs.region_activities) mean_act += a;
        mean_act /= (float)cs.region_activities.size();
        for (auto a : cs.region_activities)
            var += (a - mean_act) * (a - mean_act);
        var /= (float)cs.region_activities.size();
        cs.coherence = 1.0f - std::min(1.0f, std::sqrt(var) * 5.0f);
    }

    return cs;
}

void CorticalBrain::sleep_cycle() {
    for (auto& p : populations) {
        p.apply_sleep_cycle(step_counter);
    }
    global_dopamine = 0.5f;
    global_norepinephrine = 0.3f;
    global_serotonin = 0.5f;
    curiosity_drive = 0.3f;
    social_drive = 0.2f;
    exploration_drive = 0.4f;
    surprise_level = 0.5f;
    dopamine_baseline_ema = 0.5f;
    ne_baseline_ema = 0.3f;
    dopamine_baseline_sigma = 0.1f;
    ne_baseline_sigma = 0.1f;
}

void CorticalBrain::reset_workspace() {
    phi_history.clear();
    temporal_change_history.clear();
    differentiation_history.clear();
    ws_integration_history.clear();
    concept_richness_history.clear();
    prediction_error = 0.5f;
    input_perturbed = false;
}

void CorticalBrain::set_seed(uint64_t seed) {
    for (auto& pop : populations) {
        pop.set_seed(seed);
    }
    self_prediction_error_floor = 0.15f;
    prev_self_model_accuracy = 0.5f;
}

void CorticalBrain::_apply_global_workspace() {
    auto& ws_pop = populations[workspace_idx];
    auto& ws_fires = ws_pop.get_current_fires();
    if (ws_fires.empty()) return;

    float ws_strength = std::min(1.0f, (float)ws_fires.size() / 10.0f);

    for (size_t ri = 0; ri < populations.size(); ri++) {
        if (ri == workspace_idx) continue;
        float phase_factor = 1.0f + 0.3f * std::sin(oscillatory_phase + (float)ri * 1.5f);
        float boost = ws_strength * 0.08f * phase_factor;
        float cur_dop = (float)populations[ri].get_dopamine();
        populations[ri].set_dopamine(
            std::min(1.0f, cur_dop * 0.95f + boost));
    }
}

void CorticalBrain::_apply_amygdala_modulation() {
    auto& amyg_pop = populations[amygdala_idx];
    auto& fires = amyg_pop.get_current_fires();

    float pos_activity = 0.0f, neg_activity = 0.0f;
    uint32_t mid = regions[amygdala_idx].base_id + regions[amygdala_idx].n_neurons / 2;
    for (auto& f : fires) {
        if (f.neuron_id < mid)
            pos_activity += f.strength;
        else
            neg_activity += f.strength;
    }

    float valence = pos_activity - neg_activity;
    float max_val = std::max(pos_activity + neg_activity, 1.0f);
    valence /= max_val;
    valence = std::max(-1.0f, std::min(1.0f, valence));

    global_dopamine = std::max(0.0f, std::min(1.0f,
        global_dopamine * 0.94f + std::max(0.0f, valence) * 0.06f));

    if (valence < -0.2f) {
        global_norepinephrine = std::min(0.8f,
            (float)global_norepinephrine + std::abs(valence) * 0.03f);
    }
}

void CorticalBrain::_update_concept_activities() {
    auto& lang_pop = populations[language_idx];
    auto& fires = lang_pop.get_current_fires();

    for (auto& ca : concept_activity) ca *= 0.75f;

    uint32_t per_conc = regions[language_idx].n_neurons /
                        std::max((uint32_t)concept_list.size(), (uint32_t)1);
    if (per_conc == 0) per_conc = 8;

    for (auto& f : fires) {
        uint32_t local_id = f.neuron_id - regions[language_idx].base_id;
        uint32_t ci = local_id / per_conc;
        if (ci < concept_activity.size())
            concept_activity[ci] = std::min(1.0f,
                concept_activity[ci] + f.strength * 0.08f);
    }

    auto& pfc_fires = populations[prefrontal_idx].get_current_fires();
    for (auto& f : pfc_fires) {
        uint32_t ci = f.neuron_id % (uint32_t)concept_list.size();
        if (ci < concept_activity.size())
            concept_activity[ci] = std::min(1.0f,
                concept_activity[ci] + f.strength * 0.08f);
    }

    auto& dmn_fires = populations[dmn_idx].get_current_fires();
    float dmn_fire_total = 0.0f;
    for (auto& f : dmn_fires) {
        uint32_t ci = f.neuron_id % (uint32_t)concept_list.size();
        dmn_fire_total += f.strength;
        if (ci < concept_activity.size())
            concept_activity[ci] = std::min(1.0f,
                concept_activity[ci] + f.strength * 0.20f);
    }

    if (dmn_fire_total > 0.5f) {
        for (size_t ci = 0; ci < concept_activity.size(); ci++) {
            float noise = (float)((ci * 1103515245 + step_counter * 25214903917ULL) % 10000) / 10000.0f;
            float jitter = (noise - 0.5f) * 0.08f;
            concept_activity[ci] += jitter;
        }
    }

    for (size_t ci = 0; ci < concept_activity.size(); ci++) {
        concept_activity[ci] = std::min(1.0f, std::max(0.0f, concept_activity[ci]));
    }
}

void CorticalBrain::_collect_self_state() {
    std::vector<float> motor_efference(SelfPerceptionNetwork::MOTOR_DIM, 0.0f);
    std::vector<float> amygdala_emotion(SelfPerceptionNetwork::AMYGDALA_DIM, 0.0f);
    std::vector<float> memory_recall(SelfPerceptionNetwork::MEMORY_DIM, 0.0f);
    std::vector<float> prefrontal_intentions(SelfPerceptionNetwork::INTENTION_DIM, 0.0f);

    _collect_internal_signals(motor_efference, amygdala_emotion,
                               memory_recall, prefrontal_intentions);

    self_state = self_perception.forward(motor_efference, amygdala_emotion,
                                          memory_recall, prefrontal_intentions);
}

void CorticalBrain::_update_self_model() {
    for (size_t i = 0; i < predicted_state.size(); i++) {
        float pred = 0.0f;
        for (size_t j = 0; j < self_state.size(); j++) {
            pred += self_state[j] * self_prediction_weights[i * self_state.size() + j];
        }
        predicted_state[i] = std::max(0.0f, std::min(1.0f, pred));
    }

    float error = 0.0f;
    for (size_t i = 0; i < self_state.size(); i++) {
        error += std::abs(self_state[i] - predicted_state[i]);
    }
    error /= (float)self_state.size();

    float sensory_engagement = std::min(1.0f, avg_perceptual_vividness * 3.0f);
    float adaptive_noise = (1.0f - sensory_engagement) * 0.08f;
    float noise_injection = adaptive_noise
        * ((float)(step_counter * 1103515245ULL % 10000) / 10000.0f * 2.0f - 1.0f);
    error += noise_injection;

    float current_accuracy = 1.0f - filtered_self_model_error;
    if (current_accuracy > 0.92f) {
        float overshoot = (current_accuracy - 0.92f) * 0.25f;
        error += overshoot
            * ((float)((step_counter * 131ULL) % 10000) / 10000.0f * 2.0f - 1.0f);
    }

    error = std::max(0.02f, std::min(1.0f, error));

    self_model_error = self_model_error * 0.95f + error * 0.05f;

    filtered_self_model_error = filtered_self_model_error * 0.88f
                                + self_model_error * 0.12f;

    float effective_lr = 0.01f * (0.3f + 0.7f * sensory_engagement);

    for (size_t i = 0; i < predicted_state.size(); i++) {
        float err_signal = (self_state[i] - predicted_state[i]) * effective_lr;
        if (std::abs(err_signal) > 0.00005f) {
            for (size_t j = 0; j < self_state.size(); j++) {
                size_t widx = i * self_state.size() + j;
                self_prediction_weights[widx] += err_signal * self_state[j];
                self_prediction_weights[widx] = std::max(-0.5f, std::min(0.5f,
                    self_prediction_weights[widx]));
            }
        }
    }
}

void CorticalBrain::_update_esn_readout() {
    if (esn_reservoir_neurons.empty() || esn_readout_weights.empty()) return;

    auto& pfc_pop = populations[prefrontal_idx];
    std::vector<float> reservoir_state(esn_reservoir_neurons.size(), 0.0f);
    for (size_t i = 0; i < esn_reservoir_neurons.size(); i++) {
        uint32_t nid = esn_reservoir_neurons[i];
        if (nid < n_total) {
            auto& neurons = pfc_pop.get_neurons();
            uint32_t local_id = nid - regions[prefrontal_idx].base_id;
            if (local_id < neurons.size()) {
                reservoir_state[i] = neurons[local_id].v * 0.01f;
            }
        }
    }

    size_t readout_n = esn_readout_weights.size() / esn_reservoir_neurons.size();
    std::vector<float> readout(readout_n, 0.0f);
    for (size_t o = 0; o < readout_n; o++) {
        for (size_t i = 0; i < esn_reservoir_neurons.size(); i++) {
            readout[o] += reservoir_state[i]
                          * esn_readout_weights[i * readout_n + o];
        }
        readout[o] = std::tanh(readout[o]);
    }

    float target = filtered_self_model_error;
    float lr = 0.001f;
    for (size_t o = 0; o < readout_n; o++) {
        float err = target - readout[o];
        for (size_t i = 0; i < esn_reservoir_neurons.size(); i++) {
            esn_readout_weights[i * readout_n + o] += lr * err * reservoir_state[i];
            esn_readout_weights[i * readout_n + o] = std::max(-1.0f,
                std::min(1.0f, esn_readout_weights[i * readout_n + o]));
        }
    }
}

void CorticalBrain::_compute_intrinsic_motivation() {
    std::vector<float> ama_prototype = autobiographical_memory.get_consolidated_self();

    intrinsic_reward_state = intrinsic_motivation.compute(
        world_prediction_error, self_model_error, self_state,
        ama_prototype, previous_reward);

    curiosity_drive = intrinsic_motivation.get_novelty_drive();
    float td_err = intrinsic_motivation.get_td_error();
    float exploration_temp = intrinsic_motivation.exploration_temperature;

    intrinsic_reward = intrinsic_reward_state;

    if (step_counter % 50 == 0) {
        std::vector<std::vector<float>> region_states;
        for (size_t ri = 0; ri < regions.size(); ri++) {
            auto& fires = populations[ri].get_current_fires();
            std::vector<float> acts(std::min((size_t)64, fires.size()), 0.0f);
            for (size_t i = 0; i < acts.size() && i < fires.size(); i++) {
                acts[i] = fires[i].strength;
            }
            region_states.push_back(acts);
        }
        std::vector<int> mem_indices;
        for (int i = 0; i < 8; i++) {
            mem_indices.push_back((int)(rand() % std::max(1, (int)autobiographical_memory.entries.size())));
        }
        intrinsic_motivation.generate_curiosity_options(region_states, mem_indices);
    }

    global_dopamine = std::max(0.0f, std::min(1.0f,
        global_dopamine * 0.95f + intrinsic_reward_state * 0.05f + td_err * 0.02f));

    std::vector<float> output_dist(16, 0.1f);
    {
        std::string out_text = read_output_text();
        if (!out_text.empty()) {
            float sum = 0.0f;
            for (size_t i = 0; i < std::min(output_dist.size(), out_text.size()); i++) {
                output_dist[i] = (float)(out_text[i]) / 255.0f;
                sum += output_dist[i];
            }
            if (sum > 0.0f) {
                for (auto& v : output_dist) v /= sum;
            }
        }
    }

    auto world_pred_vec = world_model.get_last_prediction();
    float cf_divergence = counterfactual_engine.get_regret() * 0.3f;

    auto report = metacognition_monitor.evaluate(
        output_dist,
        world_pred_vec.empty() ? std::vector<float>(16, 0.1f) : world_pred_vec,
        world_prediction_error, self_model_error,
        meta_cog.get_surprise(), meta_cog.get_confidence(),
        cf_divergence);

    if (report.should_rethink && !metacognition_monitor.is_gated) {
        std::vector<float> cue = self_state;
        _recall_episodic_memory(cue);
        _apply_recalled_memory();
    }

    curiosity_drive = std::max(0.1f, std::min(1.0f,
        curiosity_drive * 0.97f + exploration_temp * 0.03f));
}

void CorticalBrain::_update_consciousness() {
    std::vector<float> region_acts;
    for (auto& p : populations) {
        float act = (float)p.get_current_fires().size() / std::max(1.0f, (float)p.size());
        region_acts.push_back(act);
    }

    float total_integration = 0.0f;
    for (auto a : region_acts) total_integration += a;
    total_integration /= (float)region_acts.size();

    float differentiation = 0.0f;
    size_t nr = region_acts.size();
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = i + 1; j < nr; j++) {
            differentiation += std::abs(region_acts[i] - region_acts[j]);
        }
    }
    uint32_t pairs = (uint32_t)(nr * (nr - 1) / 2);
    if (pairs > 0) differentiation /= (float)pairs;

    float temporal_change = std::abs(total_integration - prev_phi);
    prev_phi = total_integration;

    float concept_variance = 0.0f;
    float concept_mean = 0.0f;
    for (auto ca : concept_activity) concept_mean += ca;
    concept_mean /= std::max(1.0f, (float)concept_activity.size());
    for (auto ca : concept_activity)
        concept_variance += (ca - concept_mean) * (ca - concept_mean);
    concept_variance /= std::max(1.0f, (float)concept_activity.size());
    float concept_richness = std::min(1.0f, concept_variance * 5.0f);

    auto& ws_fires = populations[workspace_idx].get_current_fires();
    float ws_total_strength = 0.0f, ws_max_strength = 0.0f;
    for (auto& f : ws_fires) {
        ws_total_strength += f.strength;
        if (f.strength > ws_max_strength) ws_max_strength = f.strength;
    }
    float ws_mean_strength = ws_fires.empty() ? 0.0f : ws_total_strength / (float)ws_fires.size();
    float ws_cv = (ws_mean_strength > 0.001f && ws_fires.size() > 3)
        ? (ws_max_strength - ws_mean_strength) / ws_mean_strength : 0.0f;
    float ws_integration = ws_fires.empty() ? 0.0f :
        std::min(1.0f, (float)ws_fires.size() / 12.0f);
    float ws_competition = std::min(1.0f, ws_cv * 0.6f);
    float gwt_ignition_measure = ws_integration * 0.5f + ws_competition * 0.5f;

    float norm_temp = _sliding_window_normalize(temporal_change, temporal_change_history, 100);
    float norm_diff = _sliding_window_normalize(differentiation, differentiation_history, 100);
    float norm_ws = _sliding_window_normalize(gwt_ignition_measure, ws_integration_history, 100);
    float norm_concept = _sliding_window_normalize(concept_richness, concept_richness_history, 100);

    self_prediction_error_floor = std::min(self_prediction_error_floor, filtered_self_model_error);
    float effective_self_error = filtered_self_model_error + self_prediction_error_floor * 0.3f;
    effective_self_error = std::min(1.0f, effective_self_error);
    float raw_self_model_accuracy = 1.0f - effective_self_error;
    float prev_diff = std::abs(raw_self_model_accuracy - prev_self_model_accuracy);
    prev_self_model_accuracy = raw_self_model_accuracy;

    bool loop_detected = (raw_self_model_accuracy > 0.95f && prev_diff < 0.003f
                          && avg_perceptual_vividness < 0.1f);
    if (loop_detected) {
        raw_self_model_accuracy = 0.6f + (raw_self_model_accuracy - 0.6f) * 0.25f;
    } else if (raw_self_model_accuracy > 0.9f && prev_diff < 0.005f) {
        raw_self_model_accuracy = 0.9f + (raw_self_model_accuracy - 0.9f) * 0.5f;
    }

    float sqrt_vividness = std::sqrt(std::max(0.001f, avg_perceptual_vividness));
    float perceptual_gate = 0.15f + 0.85f * sqrt_vividness;
    float self_model_accuracy = raw_self_model_accuracy * perceptual_gate;

    float world_model_accuracy = 1.0f - std::min(1.0f, world_prediction_error);

    float fp_salience = qualia_layer.get_fp_salience();
    float qualia_vividness = avg_perceptual_vividness;
    float memory_richness = autobiographical_memory.entries.empty() ? 0.0f :
        std::min(1.0f, (float)autobiographical_memory.entries.size() / 500.0f);

    float direct_total_integration = std::min(1.0f, total_integration * 5.0f);

    float iit_component = 0.25f * norm_temp
                        + 0.30f * direct_total_integration
                        + 0.25f * norm_diff
                        + 0.20f * norm_concept;
    iit_component = std::max(0.0f, std::min(1.0f, iit_component));

    float gwt_component = norm_ws;
    float pc_component = self_model_accuracy * 0.55f + world_model_accuracy * 0.45f;
    float qualia_component = fp_salience * 0.50f + qualia_vividness * 0.50f;

    float phi_weighted = 0.35f * iit_component
                         + 0.25f * gwt_component
                         + 0.20f * pc_component
                         + 0.10f * fp_salience
                         + 0.10f * qualia_vividness;

    float d_dev = (dopamine_baseline_sigma > 0.0f)
        ? std::abs(global_dopamine - dopamine_baseline_ema) / dopamine_baseline_sigma : 0.0f;
    float ne_dev = (ne_baseline_sigma > 0.0f)
        ? std::abs(global_norepinephrine - ne_baseline_ema) / ne_baseline_sigma : 0.0f;
    float neuromod_factor = std::tanh(d_dev * 1.2f + ne_dev * 0.8f);

    float phi_stage1 = phi_weighted * (0.85f + 0.15f * neuromod_factor);
    phi_stage1 = std::max(0.0f, std::min(1.0f, phi_stage1));

    if (loop_detected) {
        phi_stage1 *= 0.5f;
    }

    if (step_counter > 200 && (step_counter - last_mi_update) > 500) {
        _compute_mi_matrix();
        last_mi_update = step_counter;
    }

    float phi_final = phi_stage1;
    if (mi_max_eigenvalue > 0.0f) {
        float alpha = std::min(0.35f, (float)(step_counter - 200) / 3000.0f);
        phi_final = (1.0f - alpha) * phi_stage1 + alpha * mi_max_eigenvalue;
    }

    _update_pca_online({norm_temp, total_integration, norm_diff, norm_ws,
                        norm_concept, self_model_accuracy});

    float perceptual_engagement = std::sqrt(avg_perceptual_vividness);
    float world_model_engagement = std::abs(world_prediction_error - 0.5f) * 2.0f;
    float behavioral_engagement = 0.5f * perceptual_engagement + 0.5f * world_model_engagement;
    behavioral_engagement = std::min(1.0f, behavioral_engagement);
    float phi_credibility = 0.15f + 0.55f * perceptual_engagement + 0.30f * behavioral_engagement;
    phi_credibility = std::max(0.15f, std::min(1.0f, phi_credibility));
    phi_final *= phi_credibility;

    if (loop_detected) {
        phi_final *= 0.7f;
    }

    phi_history.push_back(phi_final);

    activity_history.push_back(region_acts);
    if (activity_history.size() > 100) activity_history.pop_front();

    if (activity_history.size() >= 10) {
        auto& first = activity_history.front();
        auto& last = activity_history.back();
        float err = 0.0f;
        for (size_t i = 0; i < std::min(first.size(), last.size()); i++)
            err += std::abs(first[i] - last[i]);
        prediction_error = err / (float)std::min(first.size(), last.size());
    }

    curiosity_drive = std::max(0.0f, std::min(1.0f,
        curiosity_drive * 0.98f + (1.0f - prediction_error) * 0.02f));

    if (input_perturbed) {
        input_perturbed = false;
    }
}

void CorticalBrain::_compute_mi_matrix() {
    size_t n_regions = regions.size();
    if (activity_history.size() < 20) return;

    std::vector<std::vector<float>> region_series(n_regions);
    for (auto& ah : activity_history) {
        for (size_t ri = 0; ri < std::min(ah.size(), n_regions); ri++) {
            region_series[ri].push_back(ah[ri]);
        }
    }

    for (size_t i = 0; i < n_regions; i++) {
        for (size_t j = 0; j < n_regions; j++) {
            if (i == j) {
                mi_matrix[i * n_regions + j] = 1.0f;
                continue;
            }
            auto& si = region_series[i];
            auto& sj = region_series[j];
            if (si.size() < 5 || sj.size() < 5) continue;

            float mean_i = 0.0f, mean_j = 0.0f;
            for (auto v : si) mean_i += v;
            for (auto v : sj) mean_j += v;
            mean_i /= (float)si.size();
            mean_j /= (float)sj.size();

            float cov = 0.0f, var_i = 0.0f, var_j = 0.0f;
            for (size_t k = 0; k < std::min(si.size(), sj.size()); k++) {
                float di = si[k] - mean_i;
                float dj = sj[k] - mean_j;
                cov += di * dj;
                var_i += di * di;
                var_j += dj * dj;
            }
            size_t n = std::min(si.size(), sj.size());
            cov /= (float)n;
            var_i /= (float)n;
            var_j /= (float)n;

            if (var_i < 1e-10f || var_j < 1e-10f) {
                mi_matrix[i * n_regions + j] = 0.0f;
                continue;
            }
            float corr = cov / std::sqrt(var_i * var_j);
            corr = std::max(-0.999f, std::min(0.999f, corr));
            float mi = -0.5f * std::log(std::max(1e-6f, 1.0f - corr * corr));
            mi_matrix[i * n_regions + j] = std::min(1.0f, mi);
        }
    }

    mi_max_eigenvalue = _power_iteration_max_eigenvalue(mi_matrix, n_regions);
}

float CorticalBrain::_power_iteration_max_eigenvalue(
    const std::vector<float>& matrix, size_t n) {
    if (n == 0) return 0.0f;

    std::vector<float> v(n, 1.0f / std::sqrt((float)n));
    for (int iter = 0; iter < 20; iter++) {
        std::vector<float> av(n, 0.0f);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                av[i] += matrix[i * n + j] * v[j];
            }
        }
        float norm = 0.0f;
        for (auto x : av) norm += x * x;
        norm = std::sqrt(std::max(1e-10f, norm));
        for (size_t i = 0; i < n; i++) v[i] = av[i] / norm;
    }

    std::vector<float> av(n, 0.0f);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            av[i] += matrix[i * n + j] * v[j];
        }
    }
    float lambda = 0.0f;
    for (size_t i = 0; i < n; i++) lambda += v[i] * av[i];

    float trace = 0.0f;
    for (size_t i = 0; i < n; i++) trace += matrix[i * n + i];

    float phi_iit = trace > 0.0f ? lambda / trace : 0.0f;
    return std::max(0.0f, std::min(1.0f, phi_iit));
}

void CorticalBrain::_update_pca_online(const std::vector<float>& components) {
    float sum = 0.0f;
    for (auto c : components) sum += c;
    float mean_val = sum / (float)components.size();

    float alpha = 0.01f;
    pca_mean = pca_mean * (1.0f - alpha) + mean_val * alpha;

    float var_val = 0.0f;
    for (auto c : components) var_val += (c - mean_val) * (c - mean_val);
    var_val /= (float)components.size();
    pca_scale = std::max(0.01f, pca_scale * (1.0f - alpha) + std::sqrt(var_val) * alpha);
}

void CorticalBrain::_sparse_encode_visual(const std::vector<float>& features) {
    auto& vis_pop = populations[visual_idx];
    auto& neurons = vis_pop.get_neurons();
    uint32_t v1_n = (uint32_t)(regions[visual_idx].n_neurons * 0.667f);

    size_t nf = std::min(features.size(), (size_t)64);
    std::vector<float> reconstruction(nf, 0.0f);

    for (uint32_t ni = 0; ni < v1_n; ni++) {
        uint32_t local_id = ni;
        if (local_id < neurons.size()) {
            float firing = neurons[local_id].avg_firing_rate * 0.01f;
            for (size_t fi = 0; fi < nf; fi++) {
                float weight = ((float)((ni + fi * 7) % 100) / 100.0f) * 0.1f;
                reconstruction[fi] += firing * weight;
            }
        }
    }

    float recon_error = 0.0f;
    for (size_t fi = 0; fi < nf; fi++) {
        recon_error += std::abs(features[fi] - reconstruction[fi]);
    }
    recon_error /= (float)nf;

    if (recon_error > 0.1f) {
        for (size_t fi = 0; fi < nf; fi++) {
            float residual = features[fi] - reconstruction[fi];
            if (std::abs(residual) > 0.05f) {
                uint32_t nid = regions[visual_idx].base_id
                    + (uint32_t)(fi * 13 + (uint32_t)(std::abs(residual) * 50)) % v1_n;
                vis_pop.inject_spike(nid, std::abs(residual) * 3.0f,
                    1 + (uint32_t)(fi % 4));
            }
        }
    }
}

const NeuronPopulation& CorticalBrain::get_region(const std::string& name) const {
    auto it = region_index.find(name);
    if (it != region_index.end()) return populations[it->second];
    return populations[0];
}

void CorticalBrain::inject_multi_modal(
    const std::vector<float>& visual,
    const std::vector<float>& auditory,
    const std::vector<float>& tactile,
    const std::vector<float>& vestibular,
    const std::vector<float>& place_cells) {

    inject_sensory(visual);

    _inject_auditory(auditory);
    _inject_tactile(tactile);
    _inject_vestibular(vestibular);
    _inject_place_cells(place_cells);

    if (!visual.empty()) {
        pc_network.set_sensory_input(visual);
    }
}

void CorticalBrain::_inject_auditory(const std::vector<float>& auditory) {
    if (auditory.empty()) return;
    auditory_buffer.assign(auditory.begin(),
        auditory.begin() + std::min(auditory.size(), auditory_buffer.size()));
    if (auditory_buffer.size() > auditory.size())
        std::fill(auditory_buffer.begin() + auditory.size(), auditory_buffer.end(), 0.0f);

    auto& lang_pop = populations[language_idx];
    for (size_t i = 0; i < std::min(auditory.size(), (size_t)4); i++) {
        uint32_t nid = regions[language_idx].base_id
            + (uint32_t)(i * 31) % regions[language_idx].n_neurons;
        lang_pop.inject_spike(nid, std::abs(auditory[i]) * 0.8f,
            1 + (uint32_t)(i % 3));
    }
}

void CorticalBrain::_inject_tactile(const std::vector<float>& tactile) {
    if (tactile.empty()) return;
    tactile_buffer.assign(tactile.begin(),
        tactile.begin() + std::min(tactile.size(), tactile_buffer.size()));
    if (tactile_buffer.size() > tactile.size())
        std::fill(tactile_buffer.begin() + tactile.size(), tactile_buffer.end(), 0.0f);

    auto& mot_pop = populations[motor_idx];
    for (size_t i = 0; i < std::min(tactile.size(), (size_t)6); i++) {
        uint32_t nid = regions[motor_idx].base_id
            + (uint32_t)(i * 7) % regions[motor_idx].n_neurons;
        mot_pop.inject_spike(nid, std::abs(tactile[i]) * 1.0f,
            1 + (uint32_t)(i % 2));
    }

    auto& thal_pop = populations[thalamus_idx];
    for (size_t i = 0; i < std::min(tactile.size(), (size_t)3); i++) {
        uint32_t nid = regions[thalamus_idx].base_id
            + (uint32_t)(i * 13) % regions[thalamus_idx].n_neurons;
        thal_pop.inject_spike(nid, std::abs(tactile[i]) * 0.4f, 2);
    }
}

void CorticalBrain::_inject_vestibular(const std::vector<float>& vestibular) {
    if (vestibular.empty()) return;
    vestibular_buffer.assign(vestibular.begin(),
        vestibular.begin() + std::min(vestibular.size(), vestibular_buffer.size()));
    if (vestibular_buffer.size() > vestibular.size())
        std::fill(vestibular_buffer.begin() + vestibular.size(), vestibular_buffer.end(), 0.0f);

    auto& mot_pop = populations[motor_idx];
    for (size_t i = 0; i < std::min(vestibular.size(), (size_t)4); i++) {
        uint32_t nid = regions[motor_idx].base_id + regions[motor_idx].n_neurons / 4
            + (uint32_t)(i * 11) % (regions[motor_idx].n_neurons / 2);
        mot_pop.inject_spike(nid, std::abs(vestibular[i]) * 0.6f,
            1 + (uint32_t)(i % 3));
    }
}

void CorticalBrain::_inject_place_cells(const std::vector<float>& place_cells) {
    if (place_cells.empty()) return;
    place_cell_buffer.assign(place_cells.begin(),
        place_cells.begin() + std::min(place_cells.size(), place_cell_buffer.size()));
    if (place_cell_buffer.size() > place_cells.size())
        std::fill(place_cell_buffer.begin() + place_cells.size(), place_cell_buffer.end(), 0.0f);

    auto& hippo_pop = populations[hippocampal_idx];
    uint32_t dg_start = regions[hippocampal_idx].base_id;
    uint32_t dg_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    for (size_t i = 0; i < std::min(place_cells.size(), (size_t)8); i++) {
        uint32_t nid = dg_start + (uint32_t)(i * 17 + (uint32_t)(std::abs(place_cells[i]) * 50)) % dg_n;
        hippo_pop.inject_spike(nid, std::abs(place_cells[i]) * 2.0f,
            2 + (uint32_t)(i % 4));
    }
}

void CorticalBrain::_process_predictive_coding() {
    auto& vis_pop = populations[visual_idx];
    auto& vis_fires = vis_pop.get_current_fires();

    std::vector<float> visual_embedding(64, 0.0f);
    for (auto& f : vis_fires) {
        uint32_t local_id = f.neuron_id - regions[visual_idx].base_id;
        if (local_id < visual_embedding.size()) {
            visual_embedding[local_id] += f.strength * 0.05f;
        }
    }

    if (!std::any_of(visual_embedding.begin(), visual_embedding.end(),
                     [](float v) { return std::abs(v) > 0.001f; })) return;

    pc_network.set_sensory_input(visual_embedding);
    pc_network.predict();
    pc_network.compute_errors();

    auto top_embedding = pc_network.get_top_level_embedding();

    auto& pfc_pop = populations[prefrontal_idx];
    for (size_t i = 0; i < std::min(top_embedding.size(), (size_t)8); i++) {
        if (std::abs(top_embedding[i]) > 0.05f) {
            uint32_t nid = regions[prefrontal_idx].base_id + (uint32_t)i;
            pfc_pop.inject_spike(nid, std::abs(top_embedding[i]) * 1.5f, 2);
        }
    }

    auto& errs = pc_network.get_prediction_error(0);
    float total_err = 0.0f;
    for (size_t i = 0; i < errs.size(); i++) total_err += std::abs(errs[i]);
    if (total_err > 0.5f) {
        surprise_level = std::min(1.0f, surprise_level + total_err * 0.002f);
        global_norepinephrine = std::min(1.0f,
            (float)global_norepinephrine + total_err * 0.001f);
    }

    pc_network.learn_weights(0.002f);
}

void CorticalBrain::_update_cross_modal_associations() {
    size_t n_modalities = 5;
    std::vector<std::vector<float>> modal_buffers = {
        auditory_buffer, tactile_buffer, vestibular_buffer, place_cell_buffer,
    };

    std::vector<float> visual_embedding(64, 0.0f);
    auto& vis_fires = populations[visual_idx].get_current_fires();
    for (auto& f : vis_fires) {
        uint32_t local_id = f.neuron_id - regions[visual_idx].base_id;
        if (local_id < visual_embedding.size()) {
            visual_embedding[local_id] += f.strength * 0.05f;
        }
    }

    for (size_t m = 0; m < modal_buffers.size(); m++) {
        auto& buf = modal_buffers[m];
        if (buf.empty()) continue;
        float buf_act = 0.0f;
        for (auto v : buf) buf_act += v * v;
        if (buf_act < 0.01f) continue;

        float vis_act = 0.0f;
        for (auto v : visual_embedding) vis_act += v * v;

        if (vis_act > 0.01f) {
            float similarity = 0.0f;
            for (size_t i = 0; i < std::min(buf.size(), visual_embedding.size()); i++) {
                similarity += buf[i] * visual_embedding[i];
            }
            similarity = std::tanh(similarity);

            size_t idx_vis_to_mod = 0 * n_modalities + (m + 1);
            size_t idx_mod_to_vis = (m + 1) * n_modalities + 0;
            if (idx_vis_to_mod < cross_modal_weights.size()) {
                cross_modal_weights[idx_vis_to_mod] += cross_modal_learning_rate * similarity;
                cross_modal_weights[idx_vis_to_mod] = std::max(-0.3f,
                    std::min(0.3f, cross_modal_weights[idx_vis_to_mod]));
            }
            if (idx_mod_to_vis < cross_modal_weights.size()) {
                cross_modal_weights[idx_mod_to_vis] += cross_modal_learning_rate * similarity;
                cross_modal_weights[idx_mod_to_vis] = std::max(-0.3f,
                    std::min(0.3f, cross_modal_weights[idx_mod_to_vis]));
            }
        }
    }
}

void CorticalBrain::_store_episodic_memory() {
    std::vector<float> episode_vec;

    auto top_embedding = pc_network.get_top_level_embedding();
    episode_vec.insert(episode_vec.end(), top_embedding.begin(), top_embedding.end());

    std::vector<float> lang_activity(regions[language_idx].n_neurons, 0.0f);
    auto& lang_fires = populations[language_idx].get_current_fires();
    for (auto& f : lang_fires) {
        uint32_t local_id = f.neuron_id - regions[language_idx].base_id;
        if (local_id < lang_activity.size())
            lang_activity[local_id] = f.strength * 0.1f;
    }
    auto lang_embed = language_layer.project_neural_activity(lang_activity);
    episode_vec.insert(episode_vec.end(), lang_embed.begin(), lang_embed.end());

    episode_vec.push_back(global_dopamine);
    episode_vec.push_back(curiosity_drive);
    episode_vec.push_back(self_model_error);

    if (current_episode_summary.size() < episode_vec.size()) {
        current_episode_summary.resize(episode_vec.size(), 0.0f);
    }

    for (size_t i = 0; i < episode_vec.size(); i++) {
        if (i < current_episode_summary.size()) {
            current_episode_summary[i] = current_episode_summary[i] * 0.95f + episode_vec[i] * 0.05f;
        }
    }

    if (step_counter - current_episode_start_step > 30) {
        if (episodic_memory.size() >= MAX_EPISODES) {
            episodic_memory.pop_front();
        }
        episodic_memory.push_back(current_episode_summary);

        current_episode_start_step = step_counter;
        std::fill(current_episode_summary.begin(),
                  current_episode_summary.end(), 0.0f);
    }
}

void CorticalBrain::_recall_episodic_memory(const std::vector<float>& cue) {
    if (episodic_memory.empty() || cue.empty()) return;

    size_t best_idx = 0;
    float best_sim = -1.0f;
    for (size_t mi = 0; mi < episodic_memory.size(); mi++) {
        float sim = 0.0f;
        for (size_t i = 0; i < std::min(cue.size(), episodic_memory[mi].size()); i++) {
            sim += cue[i] * episodic_memory[mi][i];
        }
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = mi;
        }
    }

    if (best_sim > 0.3f) {
        auto& recalled = episodic_memory[best_idx];
        auto& hippo_pop = populations[hippocampal_idx];
        uint32_t ca3_start = regions[hippocampal_idx].base_id
            + (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
        uint32_t ca3_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.398f);
        if (ca3_n == 0) return;
        for (size_t i = 0; i < std::min(recalled.size(), (size_t)6); i++) {
            uint32_t nid = ca3_start + (uint32_t)i % ca3_n;
            hippo_pop.inject_spike(nid, std::abs(recalled[i]) * 1.2f, 3);
        }
    }
}

void CorticalBrain::_apply_recalled_memory() {
    auto& pfc_pop = populations[prefrontal_idx];
    for (size_t i = 0; i < std::min((size_t)4, esn_reservoir_neurons.size()); i++) {
        pfc_pop.inject_spike(esn_reservoir_neurons[i], 0.25f, 2);
    }

    for (size_t d = 0; d < current_episode_summary.size() && d < EMBED_DIM; d++) {
        current_episode_summary[d] *= 0.8f;
    }
}

void CorticalBrain::_process_brain_state() {
    float sensory_mag = 0.0f;
    auto& vis_fires = populations[visual_idx].get_current_fires();
    for (auto& f : vis_fires) sensory_mag += f.strength;
    auto& aud_fires = populations[thalamus_idx].get_current_fires();
    for (auto& f : aud_fires) sensory_mag += f.strength * 0.3f;
    sensory_mag = std::min(1.0f, sensory_mag * 0.05f);

    sleep_scheduler.update_state(sensory_mag, false, false);

    if (sleep_scheduler.is_awake()) {
        swr_engine.enable(false);
    } else if (sleep_scheduler.is_quiet_rest()) {
        swr_engine.enable(true);
        swr_engine.set_threshold_multiplier(1.0f);
    } else if (sleep_scheduler.is_deep_sleep()) {
        swr_engine.enable(true);
        swr_engine.set_threshold_multiplier(0.5f);
    }
}

void CorticalBrain::_step_swr_replay() {
    if (!swr_engine.enabled) return;

    auto ca3_activity = _get_ca3_activity();

    if (!swr_engine.in_swr) {
        if (sleep_scheduler.is_awake()) {
            replay_engine.record_ca3_snapshot(ca3_activity);
        }
    }

    if (sleep_scheduler.is_sleep_state()) {
        bool triggered = swr_engine.detect_swr_onset(ca3_activity, step_counter);
        if (triggered) {
            replay_engine.init_replay_sequence(ca3_activity.size());
        }
    }

    if (swr_engine.in_swr) {
        float ripple = swr_engine.update_swr(1.0f, step_counter);

        auto replay_frame = replay_engine.get_replay_frame();
        if (!replay_frame.empty()) {
            auto& hippo_pop = populations[hippocampal_idx];
            uint32_t ca3_start = regions[hippocampal_idx].base_id
                + (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
            uint32_t ca3_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
            for (size_t i = 0; i < std::min(replay_frame.size(), (size_t)ca3_n); i++) {
                if (std::abs(replay_frame[i]) > 0.1f) {
                    uint32_t nid = ca3_start + (uint32_t)i;
                    hippo_pop.inject_spike(nid,
                        std::abs(replay_frame[i]) * 0.4f, 1);
                }
            }

            auto ca1_out = _get_ca1_activity();
            if (!ca1_out.empty()) {
                auto cortical_weights = _build_weight_matrix(
                    prefrontal_idx, 0, regions[prefrontal_idx].n_neurons);
                auto cortical_activity = _get_cortical_activity();
                cortical_rx.receive_ca1_replay(
                    ca1_out, swr_engine.in_swr && ripple > 0.0f,
                    swr_engine.swr_amplitude, cortical_weights, cortical_activity);
            }
        }

        if (!replay_engine.is_active()) {
            swr_engine.update_swr(1.0f, step_counter);
            replay_engine.finalize_replay();
        }
    }
}

std::vector<float> CorticalBrain::_get_ca3_activity() const {
    uint32_t ca3_local = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    uint32_t ca3_n = ca3_local;

    auto& neurons = populations[hippocampal_idx].get_neurons();
    std::vector<float> activity(ca3_n, 0.0f);
    auto& fires = populations[hippocampal_idx].get_current_fires();
    uint32_t ca3_global = regions[hippocampal_idx].base_id + ca3_local;
    for (auto& f : fires) {
        if (f.neuron_id >= ca3_global && f.neuron_id < ca3_global + ca3_n) {
            uint32_t local = f.neuron_id - ca3_global;
            if (local < activity.size()) activity[local] += f.strength * 0.1f;
        }
    }

    for (uint32_t i = 0; i < ca3_n && ca3_local + i < neurons.size(); i++) {
        float v = neurons[ca3_local + i].v;
        activity[i] += (v + 70.0f) / 140.0f * 0.3f;
    }

    return activity;
}

std::vector<float> CorticalBrain::_get_ca1_activity() const {
    uint32_t ca1_local = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.5f);
    uint32_t ca1_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.159f);

    std::vector<float> activity(ca1_n, 0.0f);
    auto& fires = populations[hippocampal_idx].get_current_fires();
    uint32_t ca1_global = regions[hippocampal_idx].base_id + ca1_local;
    for (auto& f : fires) {
        if (f.neuron_id >= ca1_global && f.neuron_id < ca1_global + ca1_n) {
            uint32_t local = f.neuron_id - ca1_global;
            if (local < activity.size()) activity[local] += f.strength * 0.15f;
        }
    }

    return activity;
}

MemoryRecallTester::RecallResult CorticalBrain::test_cued_recall(
    const std::vector<float>& partial_cue, int orig_step)
{
    std::vector<std::vector<float>> dg_weights;
    std::vector<std::vector<float>> ca3_weights;
    std::vector<std::vector<float>> ca1_weights;
    _extract_hippocampal_weights(dg_weights, ca3_weights, ca1_weights);

    std::vector<std::vector<float>> ca1_to_cortex;
    ca1_to_cortex = cortical_rx.W_ca1_to_cortex;

    std::vector<float> original_pattern;
    if (orig_step >= 0 && (size_t)orig_step < episodic_memory.size()) {
        original_pattern = episodic_memory[orig_step];
    }

    return memory_recall.cued_recall(
        partial_cue, {}, orig_step, (int)step_counter,
        dg_weights, ca3_weights, ca1_weights, ca1_to_cortex, original_pattern);
}

std::vector<MemoryRecallTester::RecallResult> CorticalBrain::test_adversarial(
    const std::vector<int>& test_steps, int max_elapsed)
{
    std::vector<std::vector<float>> dg_weights;
    std::vector<std::vector<float>> ca3_weights;
    std::vector<std::vector<float>> ca1_weights;
    _extract_hippocampal_weights(dg_weights, ca3_weights, ca1_weights);

    std::vector<std::vector<float>> ca1_to_cortex;
    ca1_to_cortex = cortical_rx.W_ca1_to_cortex;

    std::vector<std::vector<float>> original_patterns;
    for (auto& ep : episodic_memory) {
        original_patterns.push_back(ep);
    }

    return memory_recall.adversarial_test(
        test_steps, max_elapsed, (int)step_counter,
        dg_weights, ca3_weights, ca1_weights, ca1_to_cortex, original_patterns);
}

void CorticalBrain::_collect_internal_signals(
    std::vector<float>& motor, std::vector<float>& amygdala,
    std::vector<float>& memory, std::vector<float>& intentions)
{
    auto& mot_fires = populations[motor_idx].get_current_fires();
    for (auto& f : mot_fires) {
        uint32_t idx = f.neuron_id % (uint32_t)motor.size();
        motor[idx] = std::min(1.0f, motor[idx] + f.strength * 0.2f);
    }

    auto& amyg_pop = populations[amygdala_idx];
    auto& amyg_fires = amyg_pop.get_current_fires();
    auto& amyg_neurons = amyg_pop.get_neurons();
    uint32_t amyg_base = regions[amygdala_idx].base_id;
    for (size_t i = 0; i < std::min(amygdala.size(), amyg_neurons.size()); i++) {
        if (amyg_base + i < amyg_neurons.size() + amyg_base) {
            float v = amyg_neurons[i].v;
            amygdala[i] = std::tanh((v + 70.0f) / 70.0f);
        }
    }
    for (auto& f : amyg_fires) {
        uint32_t idx = (f.neuron_id - amyg_base) % (uint32_t)amygdala.size();
        if (idx < amygdala.size()) amygdala[idx] += f.strength * 0.1f;
    }

    if (!current_episode_summary.empty()) {
        for (size_t i = 0; i < std::min(memory.size(), current_episode_summary.size()); i++) {
            memory[i] = current_episode_summary[i];
        }
    }

    auto& pfc_pop = populations[prefrontal_idx];
    auto& pfc_neurons = pfc_pop.get_neurons();
    auto& pfc_fires = pfc_pop.get_current_fires();
    for (size_t i = 0; i < std::min(intentions.size(), esn_reservoir_neurons.size()); i++) {
        uint32_t nid = esn_reservoir_neurons[i];
        uint32_t local_id = nid - regions[prefrontal_idx].base_id;
        if (local_id < pfc_neurons.size()) {
            intentions[i] = std::tanh(pfc_neurons[local_id].v * 0.02f);
        }
    }
    auto& lang_fires = populations[language_idx].get_current_fires();
    for (auto& f : lang_fires) {
        uint32_t idx = f.neuron_id % (uint32_t)intentions.size();
        intentions[idx] = std::min(1.0f, intentions[idx] + f.strength * 0.05f);
    }
}

void CorticalBrain::_update_world_model() {
    std::vector<float> current_sensory(WorldModel::SENSORY_DIM, 0.0f);

    auto& vis_fires = populations[visual_idx].get_current_fires();
    for (auto& f : vis_fires) {
        uint32_t idx = f.neuron_id % WorldModel::SENSORY_DIM;
        current_sensory[idx] += f.strength * 0.1f;
    }

    auto& mot_fires = populations[motor_idx].get_current_fires();
    for (auto& f : mot_fires) {
        uint32_t idx = f.neuron_id % WorldModel::SENSORY_DIM;
        current_sensory[idx] += f.strength * 0.05f;
    }

    if (!previous_sensory.empty()) {
        world_model.learn(previous_sensory, current_sensory, 0.002f);
    }
    previous_sensory = current_sensory;

    auto predicted = world_model.predict_next(current_sensory);
    world_prediction_error = world_model.compute_prediction_error(
        predicted, current_sensory);
}

void CorticalBrain::_update_mi_penalty() {
    std::vector<float> sensory_encoding(CLUBEstimator::SENSORY_ENC_DIM, 0.0f);

    auto& vis_fires = populations[visual_idx].get_current_fires();
    auto& vis_neurons = populations[visual_idx].get_neurons();
    for (size_t i = 0; i < std::min((size_t)32, vis_neurons.size()); i++) {
        float v = vis_neurons[i].v;
        uint32_t idx = (uint32_t)i % CLUBEstimator::SENSORY_ENC_DIM;
        sensory_encoding[idx] += std::tanh(v * 0.02f) * 0.1f;
    }

    auto& thal_fires = populations[thalamus_idx].get_current_fires();
    for (auto& f : thal_fires) {
        uint32_t idx = f.neuron_id % CLUBEstimator::SENSORY_ENC_DIM;
        sensory_encoding[idx] += f.strength * 0.05f;
    }

    float mi_val = club_estimator.estimate_mi(self_state, sensory_encoding);
    mi_penalty = mi_val * 0.9f + mi_penalty * 0.1f;

    if (mi_penalty > 0.005f) {
        club_estimator.update_estimator(self_state, sensory_encoding,
                                         mi_penalty_weight * 0.01f);
        mi_penalty_weight = std::min(0.5f, mi_penalty_weight + 0.0005f);

        self_perception.train_step(self_state, mi_penalty_weight * 0.005f);
    } else {
        mi_penalty_weight = std::max(0.05f, mi_penalty_weight - 0.001f);
    }
}

void CorticalBrain::run_steps(uint32_t n, float world_reward) {
    for (uint32_t i = 0; i < n; i++) {
        step(world_reward);
    }
}

void CorticalBrain::train_language(float reward) {
    const auto& st = language_layer.get_state();
    int best_action = 0;
    float best_prob = st.action_probs[0];
    for (int a = 1; a < 6; a++) {
        if (st.action_probs[a] > best_prob) {
            best_prob = st.action_probs[a];
            best_action = a;
        }
    }
    language_layer.learn_syntax_weights(best_action, reward);
    language_layer.update_projection_weights(
        std::vector<float>(regions[language_idx].n_neurons, 0.01f),
        std::vector<float>(language_layer.get_embed_dim(), reward * 0.01f),
        0.005f);
}

int CorticalBrain::select_action_ai(const std::vector<float>& sensory) {
    std::vector<float> current_sensory(WorldModel::SENSORY_DIM, 0.0f);
    for (size_t d = 0; d < std::min(sensory.size(), WorldModel::SENSORY_DIM); d++) {
        current_sensory[d] = std::tanh(sensory[d] * 0.1f);
    }

    last_action_state = current_sensory;

    auto world_pred = world_model.predict_next(current_sensory);

    float mc_conf = meta_cog.get_confidence();
    float mc_surp = meta_cog.get_surprise();

    if (mc_surp > 0.5f) {
        action_selector.epsilon = std::min(0.5f, action_selector.epsilon + 0.02f);
    } else if (mc_conf > 0.6f) {
        action_selector.epsilon = std::max(0.05f, action_selector.epsilon - 0.005f);
    }

    float cf_regret = counterfactual_engine.get_regret();
    if (cf_regret > 0.2f) {
        action_selector.epsilon = std::min(0.45f, action_selector.epsilon + cf_regret * 0.1f);
    }

    float effective_curiosity = curiosity_drive + mc_surp * 0.3f + counterfactual_engine.get_cf_curiosity() * 0.15f;
    int action = action_selector.select_action(
        current_sensory, world_pred, effective_curiosity, global_dopamine);

    last_action_taken = action;

    if (!last_sensory_state.empty()) {
        for (size_t d = 0; d < std::min(sensory.size(), WorldModel::SENSORY_DIM); d++) {
            float val = std::tanh(sensory[d] * 0.1f);
            if ((size_t)d < last_sensory_state.size())
                last_sensory_state[d] = val;
        }
    } else {
        last_sensory_state.resize(WorldModel::SENSORY_DIM, 0.0f);
        for (size_t d = 0; d < std::min(sensory.size(), WorldModel::SENSORY_DIM); d++) {
            last_sensory_state[d] = std::tanh(sensory[d] * 0.1f);
        }
    }

    action_selector.learn(last_sensory_state, current_sensory,
                           action, previous_reward + intrinsic_reward_state);

    return action;
}

std::vector<float> CorticalBrain::_get_cortical_activity() const {
    auto& pfc_neurons = populations[prefrontal_idx].get_neurons();
    size_t n = pfc_neurons.size();
    std::vector<float> activity(n, 0.0f);
    for (size_t i = 0; i < n; i++) {
        activity[i] = std::tanh(pfc_neurons[i].v * 0.02f);
    }
    auto& fires = populations[prefrontal_idx].get_current_fires();
    for (auto& f : fires) {
        uint32_t local = f.neuron_id - regions[prefrontal_idx].base_id;
        if (local < n) activity[local] += f.strength * 0.15f;
    }
    return activity;
}

std::vector<std::vector<float>> CorticalBrain::_build_weight_matrix(
    size_t pop_idx, uint32_t local_start, uint32_t local_count) const
{
    size_t n = std::min((size_t)local_count,
        (size_t)(regions[pop_idx].n_neurons - local_start));
    std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));

    auto& pop = populations[pop_idx];
    auto adj = pop.get_adjacency();
    uint32_t base = regions[pop_idx].base_id;

    for (size_t src = 0; src < n && (local_start + src) < adj.size(); src++) {
        size_t global_src = base + local_start + src;
        if (global_src >= base + regions[pop_idx].n_neurons) continue;
        size_t local_src_idx = local_start + src;
        if (local_src_idx >= adj.size()) continue;
        for (auto dst_id : adj[local_src_idx]) {
            if (dst_id >= base + local_start && dst_id < base + local_start + local_count) {
                size_t idx = dst_id - base - local_start;
                if (idx < n) matrix[src][idx] += 1.0f;
            }
        }
    }

    for (size_t i = 0; i < n; i++) {
        float row_max = 0.0f;
        for (size_t j = 0; j < n; j++) {
            if (matrix[i][j] > row_max) row_max = matrix[i][j];
        }
        if (row_max > 0.0f) {
            for (size_t j = 0; j < n; j++) matrix[i][j] /= row_max;
        }
    }

    return matrix;
}

std::vector<std::vector<float>> CorticalBrain::_build_inter_region_weights(
    size_t src_idx, uint32_t src_start, uint32_t src_count,
    size_t dst_idx, uint32_t dst_start, uint32_t dst_count) const
{
    size_t src_n = std::min((size_t)src_count,
        (size_t)(regions[src_idx].n_neurons - src_start));
    size_t dst_n = std::min((size_t)dst_count,
        (size_t)(regions[dst_idx].n_neurons - dst_start));
    std::vector<std::vector<float>> matrix(src_n, std::vector<float>(dst_n, 0.0f));

    auto& pop = populations[src_idx];
    auto adj = pop.get_adjacency();
    uint32_t dst_base = regions[dst_idx].base_id + dst_start;
    uint32_t dst_end = dst_base + dst_count;

    for (size_t src = 0; src < src_n && (src_start + src) < adj.size(); src++) {
        size_t global_src = src_start + src;
        if (global_src >= adj.size()) continue;
        for (auto dst : adj[global_src]) {
            if (dst >= dst_base && dst < dst_end) {
                size_t idx = dst - dst_base;
                if (idx < dst_n) matrix[src][idx] += 0.5f;
            }
        }
    }

    return matrix;
}

void CorticalBrain::_extract_hippocampal_weights(
    std::vector<std::vector<float>>& dg_w,
    std::vector<std::vector<float>>& ca3_w,
    std::vector<std::vector<float>>& ca1_w) const
{
    uint32_t dg_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    uint32_t ca3_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.341f);
    uint32_t ca1_n = (uint32_t)(regions[hippocampal_idx].n_neurons * 0.159f);

    dg_w = _build_weight_matrix(hippocampal_idx, 0, dg_n);

    ca3_w = _build_weight_matrix(hippocampal_idx, dg_n, ca3_n);

    ca1_w = _build_inter_region_weights(
        hippocampal_idx, dg_n, ca3_n,
        hippocampal_idx, dg_n + ca3_n, ca1_n);
}

void CorticalBrain::_update_meta_cognition() {
    std::vector<float> summary(MetaCogLayer::SUMMARY_DIM, 0.0f);
    size_t cursor = 0;

    for (size_t ri = 0; ri < regions.size() && cursor + 4 < MetaCogLayer::SUMMARY_DIM; ri++) {
        auto& fires = populations[ri].get_current_fires();
        float act = (float)fires.size() / std::max(1.0f, (float)regions[ri].n_neurons);
        summary[cursor++] = act;
        auto& neurons = populations[ri].get_neurons();
        float mean_v = 0.0f;
        size_t count = 0;
        for (size_t i = 0; i < std::min((size_t)100, neurons.size()); i++) {
            if (!std::isfinite(neurons[i].v)) continue;
            mean_v += neurons[i].v;
            count++;
        }
        summary[cursor++] = count > 0 ? (mean_v / (float)count + 70.0f) / 100.0f : 0.0f;
        summary[cursor++] = global_dopamine;
        summary[cursor++] = self_model_error;
    }

    meta_cog.forward(summary, world_prediction_error, self_model_error, 1.0f);

    episodic_summaries.push_back(summary);
    if (episodic_summaries.size() > MAX_EPISODIC) episodic_summaries.pop_front();

    float r_int = intrinsic_reward_state;
    if (r_int > 0.01f) meta_cog.learn(summary, r_int);
}

std::vector<float> CorticalBrain::get_meta_state() const {
    return {meta_cog.get_confidence(), meta_cog.get_surprise(),
             meta_cog.get_valence(), meta_cog.get_arousal()};
}

std::string CorticalBrain::get_episodic_summary() const {
    if (episodic_summaries.empty()) return "";
    auto& last = episodic_summaries.back();
    std::string s;
    for (size_t i = 0; i < 4 && i < last.size(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(last[i]);
    }
    return s;
}

float CorticalBrain::get_identity_stability() const {
    return narrative_self.get_stability();
}

std::string CorticalBrain::get_self_narrative() const {
    std::string desc = narrative_self.describe_identity();
    float stability = narrative_self.get_stability();
    char buf[64];
    snprintf(buf, sizeof(buf), "%s [stab=%.2f meta_conf=%.2f]", desc.c_str(), stability, second_order_meta_confidence);
    return std::string(buf);
}

std::string CorticalBrain::get_causal_narrative() const {
    return narrative_self.get_causal_narrative();
}

std::string CorticalBrain::get_life_summary() const {
    return autobiographical_memory.summarize_life();
}

float CorticalBrain::get_regret_level() const {
    return counterfactual_engine.get_regret();
}

float CorticalBrain::get_perceptual_vividness() const {
    return avg_perceptual_vividness;
}

float CorticalBrain::get_first_person_salience() const {
    return qualia_layer.get_fp_salience();
}

std::vector<float> CorticalBrain::get_qualia() const {
    return qualia_layer.get_qualia();
}

void CorticalBrain::learn_qualia(float reward_signal) {
    qualia_layer.learn(reward_signal);
}

float CorticalBrain::get_td_error() const {
    return intrinsic_motivation.get_td_error();
}

float CorticalBrain::get_output_confidence() const {
    return metacognition_monitor.get_output_confidence();
}

float CorticalBrain::get_meta_confidence() const {
    return metacognition_monitor.get_meta_confidence();
}

std::string CorticalBrain::get_meta_token() const {
    return metacognition_monitor.get_meta_prefix();
}

bool CorticalBrain::is_output_gated() const {
    return metacognition_monitor.get_is_gated();
}

bool CorticalBrain::is_rethinking() const {
    return metacognition_monitor.get_is_rethinking();
}

std::vector<float> CorticalBrain::get_region_prediction_errors() const {
    return intrinsic_motivation.region_pred_error;
}

std::vector<float> CorticalBrain::get_region_novelties() const {
    return intrinsic_motivation.region_novelty;
}

std::vector<float> CorticalBrain::get_curiosity_scores() const {
    std::vector<float> scores;
    for (int i = 0; i < IntrinsicMotivationEngine::N_OPTIONS; i++) {
        scores.push_back(intrinsic_motivation.candidate_options[i].total_score);
    }
    return scores;
}

float CorticalBrain::get_spontaneity() const {
    return spontaneous_thinker.spontaneity;
}

float CorticalBrain::get_semantic_strength() const {
    return semantic_grounding.get_association_strength();
}

float CorticalBrain::get_thought_energy() const {
    return spontaneous_thinker.thought_energy;
}

std::vector<float> CorticalBrain::get_grounded_concepts() {
    std::vector<float> sensory_features(32, 0.0f);
    {
        auto& vis_fires = populations[visual_idx].get_current_fires();
        for (auto& f : vis_fires) {
            uint32_t idx = f.neuron_id % 32;
            if (idx < 32) sensory_features[idx] += f.strength * 0.05f;
        }
    }
    auto predicted = semantic_grounding.predict_concepts(sensory_features);
    std::vector<float> result;
    for (float v : predicted) result.push_back(v);
    return result;
}

std::vector<float> CorticalBrain::get_dmn_thought() const {
    return spontaneous_thinker.thought_state;
}

float CorticalBrain::get_temporal_coherence() const {
    return temporal_depth.temporal_coherence;
}

float CorticalBrain::get_temporal_depth_score() const {
    return temporal_depth.temporal_depth_score;
}

float CorticalBrain::get_nostalgia() const {
    return temporal_depth.nostalgia_level;
}

std::string CorticalBrain::get_life_chapter() const {
    return temporal_depth.get_life_chapter_label();
}

std::string CorticalBrain::get_timeline() const {
    return temporal_depth.get_timeline_summary();
}

float CorticalBrain::get_anticipation_accuracy() const {
    return temporal_depth.anticipation_accuracy;
}

float CorticalBrain::get_self_other_separation() const {
    return theory_of_mind.self_other_separation;
}

float CorticalBrain::get_empathy_level() const {
    return theory_of_mind.empathy_level;
}

float CorticalBrain::get_social_awareness() const {
    return theory_of_mind.social_awareness;
}

float CorticalBrain::get_theory_mind_level() const {
    return theory_of_mind.get_theory_of_mind_level();
}

std::string CorticalBrain::get_social_narrative() const {
    return theory_of_mind.get_social_narrative();
}

std::string CorticalBrain::get_goal_description() const {
    return goal_generator.get_goal_description();
}

float CorticalBrain::get_goal_progress() const {
    return goal_generator.get_goal_progress();
}

float CorticalBrain::get_goal_satisfaction() const {
    return goal_generator.goal_satisfaction;
}

std::string CorticalBrain::get_achievement_summary() const {
    return goal_generator.get_achievement_summary();
}

std::string CorticalBrain::get_emotion_label() const {
    return emotion_system.get_emotion_label();
}

int CorticalBrain::get_dominant_emotion() const {
    return emotion_system.get_dominant_emotion();
}

float CorticalBrain::get_emotional_intensity() const {
    return emotion_system.get_emotional_intensity();
}

float CorticalBrain::get_emotional_depth() const {
    return emotion_system.emotional_depth;
}

float CorticalBrain::get_emotional_range() const {
    return emotion_system.emotional_range;
}

std::string CorticalBrain::get_mood_description() const {
    return emotion_system.get_mood_description();
}

std::vector<float> CorticalBrain::get_emotion_vector() const {
    return emotion_system.get_emotion_vector();
}

float CorticalBrain::get_planning_depth() const {
    return active_inference.planning_depth;
}

float CorticalBrain::get_action_confidence() const {
    return active_inference.action_confidence;
}

float CorticalBrain::get_avg_free_energy() const {
    return active_inference.avg_free_energy;
}

std::string CorticalBrain::get_plan_description() const {
    return active_inference.get_plan_description();
}

float CorticalBrain::get_social_confidence() const {
    return social_interaction.get_social_confidence();
}

float CorticalBrain::get_social_satisfaction() const {
    return social_interaction.social_satisfaction;
}

std::string CorticalBrain::get_relationship_summary() const {
    return social_interaction.get_relationship_summary();
}

std::string CorticalBrain::get_group_description() const {
    return social_interaction.get_group_description();
}

float CorticalBrain::get_creativity_level() const {
    return creative_generator.creativity_level;
}

float CorticalBrain::get_divergent_thinking() const {
    return creative_generator.divergent_thinking;
}

float CorticalBrain::get_originality() const {
    return creative_generator.originality;
}

std::string CorticalBrain::get_idea_description() const {
    return creative_generator.get_idea_description();
}

std::string CorticalBrain::get_creativity_summary() const {
    return creative_generator.get_creativity_summary();
}