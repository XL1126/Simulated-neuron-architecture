#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "neuron_population.h"
#include "stdp_engine.h"
#include "cortical_brain.h"
#include "sleep_wake_scheduler.h"
#include "swr_engine.h"
#include "memory_recall.h"
#include "spa_neural_mapper.h"
#include "global_workspace.h"

namespace py = pybind11;

PYBIND11_MODULE(core_cpp, m) {
    m.doc() = "SNA Cortex - Structured innate neural architecture";

    py::class_<STDPConfig>(m, "STDPConfig")
        .def(py::init<>())
        .def_readwrite("a_plus", &STDPConfig::a_plus)
        .def_readwrite("a_minus", &STDPConfig::a_minus)
        .def_readwrite("tau_plus_ms", &STDPConfig::tau_plus_ms)
        .def_readwrite("tau_minus_ms", &STDPConfig::tau_minus_ms)
        .def_readwrite("dopamine_k", &STDPConfig::dopamine_k)
        .def_readwrite("history_window_ms", &STDPConfig::history_window_ms);

    py::class_<SpikeFireEvent>(m, "SpikeFireEvent")
        .def(py::init<>())
        .def_readwrite("neuron_id", &SpikeFireEvent::neuron_id)
        .def_readwrite("strength", &SpikeFireEvent::strength);

    py::class_<NeuronPopulation>(m, "NeuronPopulation")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("num_neurons"), py::arg("avg_degree") = 20)
        .def("update", &NeuronPopulation::update,
             py::arg("step"), py::arg("noise_level"), py::arg("dopamine"))
        .def("inject_spike", &NeuronPopulation::inject_spike,
             py::arg("target_id"), py::arg("strength"), py::arg("delay_ms") = 1)
        .def("inject_spike_group", &NeuronPopulation::inject_spike_group,
             py::arg("target_ids"), py::arg("strength"), py::arg("delay_ms") = 1)
        .def("set_dopamine", &NeuronPopulation::set_dopamine, py::arg("value"))
        .def("get_dopamine", &NeuronPopulation::get_dopamine)
        .def("add_synapse", &NeuronPopulation::add_synapse,
             py::arg("src"), py::arg("dst"), py::arg("weight"), py::arg("delay"))
        .def("add_synaptic_current", &NeuronPopulation::add_synaptic_current,
             py::arg("neuron_id"), py::arg("current"))
        .def("set_neuron_bias", &NeuronPopulation::set_neuron_bias,
             py::arg("neuron_id"), py::arg("bias"))
        .def("get_current_fires", &NeuronPopulation::get_current_fires)
        .def("get_neurons", &NeuronPopulation::get_neurons)
        .def("apply_sleep_cycle", &NeuronPopulation::apply_sleep_cycle,
             py::arg("step"))
        .def("size", &NeuronPopulation::size)
        .def("set_stdp_config", &NeuronPopulation::set_stdp_config,
             py::arg("config"))
        .def("get_stdp_config", &NeuronPopulation::get_stdp_config)
        .def("set_seed", &NeuronPopulation::set_seed,
             py::arg("seed"))
        .def("update_eligibility_traces", &NeuronPopulation::update_eligibility_traces)
        .def("apply_credit", &NeuronPopulation::apply_credit,
             py::arg("reward"), py::arg("eta"))
        .def("decay_eligibility_traces", &NeuronPopulation::decay_eligibility_traces,
             py::arg("lambda"))
        .def("connect_random", &NeuronPopulation::connect_random,
             py::arg("prob"), py::arg("max_per_neuron"))
        .def("build_small_world", &NeuronPopulation::build_small_world,
             py::arg("group_size"), py::arg("local_prob"), py::arg("long_range_prob"))
        .def("build_competitive_pool", &NeuronPopulation::build_competitive_pool,
             py::arg("start_idx"), py::arg("end_idx"))
        .def("build_erdos_renyi", &NeuronPopulation::build_erdos_renyi,
             py::arg("start_idx"), py::arg("end_idx"), py::arg("prob"));

    py::class_<BrainRegion>(m, "BrainRegion")
        .def_readonly("name", &BrainRegion::name)
        .def_readonly("base_id", &BrainRegion::base_id)
        .def_readonly("n_neurons", &BrainRegion::n_neurons)
        .def_readonly("n_actions", &BrainRegion::n_actions)
        .def_readonly("n_concepts", &BrainRegion::n_concepts);

    py::class_<ConsciousnessState>(m, "ConsciousnessState")
        .def_readonly("phi", &ConsciousnessState::phi)
        .def_readonly("phi_base", &ConsciousnessState::phi_base)
        .def_readonly("phi_surprise", &ConsciousnessState::phi_surprise)
        .def_readonly("global_ignition", &ConsciousnessState::global_ignition)
        .def_readonly("self_prediction_error",
                      &ConsciousnessState::self_prediction_error)
        .def_readonly("coherence", &ConsciousnessState::coherence)
        .def_readonly("attention_focus", &ConsciousnessState::attention_focus)
        .def_readonly("region_activities",
                      &ConsciousnessState::region_activities)
        .def_readonly("active_concepts",
                      &ConsciousnessState::active_concepts);

    py::class_<SWREngine::SWRStats>(m, "SWRStats")
        .def_readonly("count", &SWREngine::SWRStats::count)
        .def_readonly("mean_duration", &SWREngine::SWRStats::mean_duration)
        .def_readonly("mean_amplitude", &SWREngine::SWRStats::mean_amplitude)
        .def_readonly("replay_coverage", &SWREngine::SWRStats::replay_coverage);

    py::class_<MemoryRecallTester::RecallResult>(m, "RecallResult")
        .def_readonly("success", &MemoryRecallTester::RecallResult::success)
        .def_readonly("confidence", &MemoryRecallTester::RecallResult::confidence)
        .def_readonly("steps_since_encoding",
                      &MemoryRecallTester::RecallResult::steps_since_encoding)
        .def_readonly("reconstruction_error",
                      &MemoryRecallTester::RecallResult::reconstruction_error)
        .def_readonly("recalled_pattern",
                      &MemoryRecallTester::RecallResult::recalled_pattern);

    py::enum_<SleepWakeScheduler::SystemState>(m, "BrainState")
        .value("AWAKE", SleepWakeScheduler::SystemState::AWAKE_ONLINE)
        .value("QUIET_REST", SleepWakeScheduler::SystemState::QUIET_REST)
        .value("DEEP_SLEEP", SleepWakeScheduler::SystemState::DEEP_SLEEP)
        .export_values();

    py::class_<CorticalBrain>(m, "CorticalBrain")
        .def(py::init<uint32_t, std::vector<std::string>>(),
             py::arg("total_neurons"), py::arg("concepts"))
        .def("step", &CorticalBrain::step,
             py::arg("world_reward") = 0.0f)
        .def("run_steps", &CorticalBrain::run_steps,
             py::arg("n"), py::arg("world_reward") = 0.0f)
        .def("inject_sensory", &CorticalBrain::inject_sensory,
             py::arg("features"))
        .def("inject_multi_modal", &CorticalBrain::inject_multi_modal,
             py::arg("visual"), py::arg("auditory"),
             py::arg("tactile"), py::arg("vestibular"),
             py::arg("place_cells"))
        .def("inject_text", &CorticalBrain::inject_text,
             py::arg("text"))
        .def("inject_reward", &CorticalBrain::inject_reward,
             py::arg("reward"))
        .def("read_motor_output", &CorticalBrain::read_motor_output)
        .def("read_thought_vector", &CorticalBrain::read_thought_vector)
        .def("read_output_text", &CorticalBrain::read_output_text)
        .def("read_consciousness", &CorticalBrain::read_consciousness)
        .def("sleep_cycle", &CorticalBrain::sleep_cycle)
        .def("reset_workspace", &CorticalBrain::reset_workspace)
        .def("total_neurons", &CorticalBrain::total_neurons)
        .def("get_regions", &CorticalBrain::get_regions)
        .def("force_sleep", &CorticalBrain::force_sleep)
        .def("force_awake", &CorticalBrain::force_awake)
        .def("set_seed", &CorticalBrain::set_seed, py::arg("seed"))
        .def("get_brain_state", &CorticalBrain::get_brain_state)
        .def("get_swr_stats", &CorticalBrain::get_swr_stats)
        .def("get_replay_progress", &CorticalBrain::get_replay_progress)
        .def("is_replaying", &CorticalBrain::is_replaying)
        .def("test_cued_recall", &CorticalBrain::test_cued_recall,
             py::arg("partial_cue"), py::arg("orig_step"))
        .def("test_adversarial", &CorticalBrain::test_adversarial,
             py::arg("test_steps"), py::arg("max_elapsed"))
        .def("train_language", &CorticalBrain::train_language,
             py::arg("reward"))
        .def("select_action_ai", &CorticalBrain::select_action_ai,
             py::arg("sensory"))
        .def("get_meta_state", &CorticalBrain::get_meta_state)
        .def("get_episodic_summary", &CorticalBrain::get_episodic_summary)
        .def("get_identity_stability", &CorticalBrain::get_identity_stability)
        .def("get_self_narrative", &CorticalBrain::get_self_narrative)
        .def("get_causal_narrative", &CorticalBrain::get_causal_narrative)
        .def("get_life_summary", &CorticalBrain::get_life_summary)
        .def("get_regret_level", &CorticalBrain::get_regret_level)
        .def("get_perceptual_vividness", &CorticalBrain::get_perceptual_vividness)
        .def("get_first_person_salience", &CorticalBrain::get_first_person_salience)
        .def("get_qualia", &CorticalBrain::get_qualia)
        .def("learn_qualia", &CorticalBrain::learn_qualia, py::arg("reward_signal"))
        .def("get_td_error", &CorticalBrain::get_td_error)
        .def("get_output_confidence", &CorticalBrain::get_output_confidence)
        .def("get_meta_confidence", &CorticalBrain::get_meta_confidence)
        .def("get_meta_token", &CorticalBrain::get_meta_token)
        .def("is_output_gated", &CorticalBrain::is_output_gated)
        .def("is_rethinking", &CorticalBrain::is_rethinking)
        .def("get_region_prediction_errors", &CorticalBrain::get_region_prediction_errors)
        .def("get_region_novelties", &CorticalBrain::get_region_novelties)
        .def("get_curiosity_scores", &CorticalBrain::get_curiosity_scores)
        .def("get_spontaneity", &CorticalBrain::get_spontaneity)
        .def("get_semantic_strength", &CorticalBrain::get_semantic_strength)
        .def("get_thought_energy", &CorticalBrain::get_thought_energy)
        .def("get_grounded_concepts", &CorticalBrain::get_grounded_concepts)
        .def("get_dmn_thought", &CorticalBrain::get_dmn_thought)
        .def("get_temporal_coherence", &CorticalBrain::get_temporal_coherence)
        .def("get_temporal_depth_score", &CorticalBrain::get_temporal_depth_score)
        .def("get_nostalgia", &CorticalBrain::get_nostalgia)
        .def("get_life_chapter", &CorticalBrain::get_life_chapter)
        .def("get_timeline", &CorticalBrain::get_timeline)
        .def("get_anticipation_accuracy", &CorticalBrain::get_anticipation_accuracy)
        .def("get_self_other_separation", &CorticalBrain::get_self_other_separation)
        .def("get_empathy_level", &CorticalBrain::get_empathy_level)
        .def("get_social_awareness", &CorticalBrain::get_social_awareness)
        .def("get_theory_mind_level", &CorticalBrain::get_theory_mind_level)
        .def("get_social_narrative", &CorticalBrain::get_social_narrative)
        .def("get_goal_description", &CorticalBrain::get_goal_description)
        .def("get_goal_progress", &CorticalBrain::get_goal_progress)
        .def("get_goal_satisfaction", &CorticalBrain::get_goal_satisfaction)
        .def("get_achievement_summary", &CorticalBrain::get_achievement_summary)
        .def("get_emotion_label", &CorticalBrain::get_emotion_label)
        .def("get_dominant_emotion", &CorticalBrain::get_dominant_emotion)
        .def("get_emotional_intensity", &CorticalBrain::get_emotional_intensity)
        .def("get_emotional_depth", &CorticalBrain::get_emotional_depth)
        .def("get_emotional_range", &CorticalBrain::get_emotional_range)
        .def("get_mood_description", &CorticalBrain::get_mood_description)
        .def("get_emotion_vector", &CorticalBrain::get_emotion_vector)
        .def("get_planning_depth", &CorticalBrain::get_planning_depth)
        .def("get_action_confidence", &CorticalBrain::get_action_confidence)
        .def("get_avg_free_energy", &CorticalBrain::get_avg_free_energy)
        .def("get_plan_description", &CorticalBrain::get_plan_description)
        .def("get_social_confidence", &CorticalBrain::get_social_confidence)
        .def("get_social_satisfaction", &CorticalBrain::get_social_satisfaction)
        .def("get_relationship_summary", &CorticalBrain::get_relationship_summary)
        .def("get_group_description", &CorticalBrain::get_group_description)
        .def("get_creativity_level", &CorticalBrain::get_creativity_level)
        .def("get_divergent_thinking", &CorticalBrain::get_divergent_thinking)
        .def("get_originality", &CorticalBrain::get_originality)
        .def("get_idea_description", &CorticalBrain::get_idea_description)
        .def("get_creativity_summary", &CorticalBrain::get_creativity_summary);

    py::class_<GWBroadcastEvent>(m, "GWBroadcastEvent")
        .def(py::init<>())
        .def_readwrite("target_neuron_id", &GWBroadcastEvent::target_neuron_id)
        .def_readwrite("strength", &GWBroadcastEvent::strength)
        .def_readwrite("delay_ms", &GWBroadcastEvent::delay_ms);

    py::class_<SPANeuralMapper>(m, "SPANeuralMapper")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("total_neurons"), py::arg("dim"),
             py::arg("neurons_per_concept"))
        .def("add_concept", &SPANeuralMapper::add_concept,
             py::arg("concept"), py::arg("force_new") = false)
        .def("get_concept_neurons", &SPANeuralMapper::get_concept_neurons,
             py::arg("concept"))
        .def("map_vector_to_spikes", [](SPANeuralMapper& self,
             const std::vector<float>& vec, float energy_scale) {
            std::vector<std::pair<uint32_t, float>> out;
            self.map_vector_to_spikes(vec, out, energy_scale);
            return out;
        }, py::arg("vec"), py::arg("energy_scale") = 1.0f)
        .def("map_spikes_to_vector", &SPANeuralMapper::map_spikes_to_vector,
             py::arg("active_spikes"))
        .def("learn_from_pair", &SPANeuralMapper::learn_from_pair,
             py::arg("pre_ids"), py::arg("post_ids"), py::arg("reward"))
        .def("reinforce_concept", &SPANeuralMapper::reinforce_concept,
             py::arg("concept"), py::arg("strength"))
        .def("decay_eligibility", &SPANeuralMapper::decay_eligibility,
             py::arg("rate"))
        .def("get_concept_vector", &SPANeuralMapper::get_concept_vector,
             py::arg("concept"))
        .def("resolve_neuron", &SPANeuralMapper::resolve_neuron,
             py::arg("neuron_id"));

    py::class_<GlobalWorkspacePopulation>(m, "GlobalWorkspacePopulation")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("num_neurons"), py::arg("competition_window_ms") = 10)
        .def("step", &GlobalWorkspacePopulation::step,
             py::arg("inputs"), py::arg("dopamine") = 0.0f)
        .def("broadcast", &GlobalWorkspacePopulation::broadcast,
             py::arg("target_population_size"))
        .def("get_phi_estimate", &GlobalWorkspacePopulation::get_phi_estimate)
        .def("is_ignited", &GlobalWorkspacePopulation::is_ignited)
        .def("get_ignition_strength", &GlobalWorkspacePopulation::get_ignition_strength)
        .def("get_activity", &GlobalWorkspacePopulation::get_activity_level)
        .def("get_competition_entropy", &GlobalWorkspacePopulation::get_competition_entropy)
        .def("get_winner_margin", &GlobalWorkspacePopulation::get_winner_margin)
        .def("get_fatigue_level", &GlobalWorkspacePopulation::get_fatigue_level)
        .def("get_saturation_risk", &GlobalWorkspacePopulation::get_ignition_saturation_risk)
        .def("get_winner_diversity", &GlobalWorkspacePopulation::get_winner_diversity)
        .def("size", &GlobalWorkspacePopulation::size)
        .def("reset", &GlobalWorkspacePopulation::reset)
        .def("set_seed", &GlobalWorkspacePopulation::set_seed, py::arg("seed"));
}