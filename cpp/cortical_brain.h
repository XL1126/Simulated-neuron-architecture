#pragma once
#include "neuron_population.h"
#include "innate_circuits.h"
#include "language_layer.h"
#include "predictive_coding_layer.h"
#include "swr_engine.h"
#include "replay_engine.h"
#include "cortical_reactivation.h"
#include "memory_recall.h"
#include "sleep_wake_scheduler.h"
#include "self_perception_network.h"
#include "world_model.h"
#include "synapse_buffer.h"
#include "fast_math.h"
#include "action_selector.h"
#include "meta_cog.h"
#include "intrinsic_value.h"
#include "state_word_mapper.h"
#include "narrative_self.h"
#include "language_sequence.h"
#include "autobiographical_memory.h"
#include "counterfactual_engine.h"
#include "qualia_layer.h"
#include "intrinsic_motivation_engine.h"
#include "metacognition_monitor.h"
#include "spontaneous_thinker.h"
#include "semantic_grounding.h"
#include "temporal_depth.h"
#include "theory_of_mind.h"
#include "goal_generator.h"
#include "emotion_system.h"
#include "active_inference_engine.h"
#include "social_interaction.h"
#include "creative_generator.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <set>
#include <valarray>

struct BrainRegion {
    std::string name;
    uint32_t base_id;
    uint32_t n_neurons;
    uint32_t n_actions;
    uint32_t n_concepts;
};

struct ConsciousnessState {
    float phi;
    float global_ignition;
    float self_prediction_error;
    float coherence;
    float attention_focus;
    float phi_base;
    float phi_surprise;
    std::vector<float> region_activities;
    std::vector<std::string> active_concepts;
};

class CorticalBrain {
public:
    CorticalBrain(uint32_t total_neurons, const std::vector<std::string>& concepts);

    void step(float world_reward = 0.0f);
    void run_steps(uint32_t n, float world_reward = 0.0f);
    void inject_sensory(const std::vector<float>& features);
    void inject_multi_modal(const std::vector<float>& visual,
                            const std::vector<float>& auditory,
                            const std::vector<float>& tactile,
                            const std::vector<float>& vestibular,
                            const std::vector<float>& place_cells);
    void inject_text(const std::string& text);
    void inject_reward(float reward);

    std::vector<uint32_t> read_motor_output() const;
    std::vector<float> read_thought_vector() const;
    std::string read_output_text() const;
    ConsciousnessState read_consciousness() const;

    void sleep_cycle();
    void reset_workspace();
    size_t total_neurons() const { return n_total; }

    void force_sleep() { sleep_scheduler.update_state(0.0f, true, false); }
    void force_awake() { sleep_scheduler.update_state(0.0f, false, true); }
    void set_seed(uint64_t seed);
    SleepWakeScheduler::SystemState get_brain_state() const { return sleep_scheduler.get_state(); }
    SWREngine::SWRStats get_swr_stats() const { return swr_engine.compute_stats(); }
    int get_replay_progress() const { return replay_engine.current_pos(); }
    bool is_replaying() const { return replay_engine.is_active(); }

    void train_language(float reward);

    int select_action_ai(const std::vector<float>& sensory);

    std::vector<float> get_meta_state() const;
    std::string get_episodic_summary() const;
    float get_identity_stability() const;
    std::string get_self_narrative() const;
    std::string get_causal_narrative() const;
    std::string get_life_summary() const;
    float get_regret_level() const;
    float get_perceptual_vividness() const;
    float get_first_person_salience() const;
    std::vector<float> get_qualia() const;
    void learn_qualia(float reward_signal);

    float get_td_error() const;
    float get_output_confidence() const;
    float get_meta_confidence() const;
    std::string get_meta_token() const;
    bool is_output_gated() const;
    bool is_rethinking() const;
    std::vector<float> get_region_prediction_errors() const;
    std::vector<float> get_region_novelties() const;
    std::vector<float> get_curiosity_scores() const;
    float get_spontaneity() const;
    float get_semantic_strength() const;
    float get_thought_energy() const;
    std::vector<float> get_grounded_concepts();
    std::vector<float> get_dmn_thought() const;

    float get_temporal_coherence() const;
    float get_temporal_depth_score() const;
    float get_nostalgia() const;
    std::string get_life_chapter() const;
    std::string get_timeline() const;
    float get_anticipation_accuracy() const;

    float get_self_other_separation() const;
    float get_empathy_level() const;
    float get_social_awareness() const;
    float get_theory_mind_level() const;
    std::string get_social_narrative() const;

    std::string get_goal_description() const;
    float get_goal_progress() const;
    float get_goal_satisfaction() const;
    std::string get_achievement_summary() const;

    std::string get_emotion_label() const;
    int get_dominant_emotion() const;
    float get_emotional_intensity() const;
    float get_emotional_depth() const;
    float get_emotional_range() const;
    std::string get_mood_description() const;
    std::vector<float> get_emotion_vector() const;

    float get_planning_depth() const;
    float get_action_confidence() const;
    float get_avg_free_energy() const;
    std::string get_plan_description() const;

    float get_social_confidence() const;
    float get_social_satisfaction() const;
    std::string get_relationship_summary() const;
    std::string get_group_description() const;

    float get_creativity_level() const;
    float get_divergent_thinking() const;
    float get_originality() const;
    std::string get_idea_description() const;
    std::string get_creativity_summary() const;

    MemoryRecallTester::RecallResult test_cued_recall(
        const std::vector<float>& partial_cue, int orig_step);
    std::vector<MemoryRecallTester::RecallResult> test_adversarial(
        const std::vector<int>& test_steps, int max_elapsed);

    const NeuronPopulation& get_region(const std::string& name) const;
    const std::vector<BrainRegion>& get_regions() const { return regions; }

private:
    uint32_t n_total;
    std::vector<NeuronPopulation> populations;
    std::vector<BrainRegion> regions;
    std::unordered_map<std::string, size_t> region_index;

    InnateCircuitBuilder circuit_builder;
    std::vector<std::string> concept_list;
    std::vector<float> concept_activity;
    std::vector<float> drive_activations;
    uint64_t step_counter;

    float global_dopamine;
    float global_serotonin;
    float global_norepinephrine;
    float prediction_error;
    float prev_phi;
    float previous_reward;
    float curiosity_drive;
    float social_drive;
    float exploration_drive;
    float surprise_level;
    float oscillatory_phase;
    float intrinsic_reward;
    float filtered_self_model_error;
    std::deque<float> phi_history;
    std::deque<std::vector<float>> activity_history;
    std::deque<float> temporal_change_history;
    std::deque<float> differentiation_history;
    std::deque<float> ws_integration_history;
    std::deque<float> concept_richness_history;

    float dopamine_baseline_ema;
    float ne_baseline_ema;
    float dopamine_baseline_sigma;
    float ne_baseline_sigma;

    std::vector<float> mi_matrix;
    float mi_max_eigenvalue;
    uint64_t last_mi_update;

    float pca_mean;
    float pca_scale;

    std::vector<float> self_state;
    std::vector<float> predicted_state;
    std::vector<float> self_prediction_weights;
    float self_model_error;

    std::vector<float> esn_readout_weights;
    std::vector<uint32_t> esn_reservoir_neurons;
    static constexpr size_t ESN_RESERVOIR_SIZE = 256;
    static constexpr size_t ESN_READOUT_SIZE = 32;

    bool input_perturbed;

    LanguageLayer language_layer;
    PredictiveCodingNetwork pc_network;
    static constexpr size_t PC_N_LEVELS = 3;
    static constexpr size_t EMBED_DIM = 32;

    SWREngine swr_engine;
    ReplayEngine replay_engine;
    CorticalReactivationReceiver cortical_rx;
    MemoryRecallTester memory_recall;
    SleepWakeScheduler sleep_scheduler;

    SelfPerceptionNetwork self_perception;
    WorldModel world_model;
    CLUBEstimator club_estimator;
    std::vector<float> fast_exp_table;

    float world_prediction_error;
    float mi_penalty;
    float mi_penalty_weight;
    std::vector<float> previous_sensory;

    float global_activity_ema;
    float adaptive_noise_boost;

    ActionSelector action_selector;
    std::vector<float> last_sensory_state;
    int last_action_taken;
    std::vector<float> last_action_state;

    MetaCogLayer meta_cog;
    IntrinsicValueNucleus intrinsic_value;
    StateToWordMapper state_word_mapper;
    std::deque<std::vector<float>> episodic_summaries;
    static constexpr size_t MAX_EPISODIC = 128;

    float intrinsic_reward_state;

    NarrativeSelf narrative_self;
    LanguageSequenceGenerator lang_seq_gen;

    std::deque<float> second_order_conf;
    float second_order_meta_confidence;

    AutobiographicalMemory autobiographical_memory;
    CounterfactualEngine counterfactual_engine;
    QualiaLayer qualia_layer;
    IntrinsicMotivationEngine intrinsic_motivation;
    MetacognitionMonitor metacognition_monitor;
    SpontaneousThinker spontaneous_thinker;
    SemanticGrounding semantic_grounding;
    TemporalDepth temporal_depth;
    TheoryOfMind theory_of_mind;
    GoalGenerator goal_generator;

    EmotionSystem emotion_system;
    ActiveInferenceEngine active_inference;
    SocialInteractionModel social_interaction;
    CreativeGenerator creative_generator;

    std::deque<float> qualia_vividness_history;
    float avg_perceptual_vividness;

    float self_prediction_error_floor;
    float prev_self_model_accuracy;
    mutable float prev_global_ignition_cache;

    static constexpr size_t CROSS_MODAL_DIM = 32;

    std::vector<float> auditory_buffer;
    std::vector<float> tactile_buffer;
    std::vector<float> vestibular_buffer;
    std::vector<float> place_cell_buffer;

    std::vector<float> cross_modal_weights;
    float cross_modal_learning_rate;
    std::deque<std::vector<float>> episodic_memory;
    static constexpr size_t MAX_EPISODES = 50;
    size_t current_episode_start_step;
    std::vector<float> current_episode_summary;

    size_t visual_idx, motor_idx, hippocampal_idx, prefrontal_idx;
    size_t amygdala_idx, language_idx, workspace_idx;
    size_t thalamus_idx, claustrum_idx, dmn_idx;

    void _wire_all_regions();
    void _wire_region(const BrainRegion& reg);
    void _wire_region_interconnects();
    void _add_recurrent_self_excitation();
    void _apply_oscillatory_coupling();
    void _update_drives();
    void _update_self_model();
    void _update_world_model();
    void _update_mi_penalty();
    void _update_meta_cognition();
    void _compute_intrinsic_motivation();
    void _update_consciousness();
    void _apply_global_workspace();
    void _apply_amygdala_modulation();
    void _update_concept_activities();
    void _collect_self_state();
    void _collect_internal_signals(std::vector<float>& motor,
                                    std::vector<float>& amygdala,
                                    std::vector<float>& memory,
                                    std::vector<float>& intentions);
    void _setup_esn_prefrontal();
    void _update_esn_readout();
    void _update_esn_readout_online(float lr);
    float _sliding_window_normalize(float value, std::deque<float>& history, size_t max_size);
    void _update_neuromodulator_baselines();
    void _compute_mi_matrix();
    float _power_iteration_max_eigenvalue(const std::vector<float>& matrix, size_t n);
    void _update_pca_online(const std::vector<float>& components);
    void _sparse_encode_visual(const std::vector<float>& features);
    void _inject_auditory(const std::vector<float>& auditory);
    void _inject_tactile(const std::vector<float>& tactile);
    void _inject_vestibular(const std::vector<float>& vestibular);
    void _inject_place_cells(const std::vector<float>& place_cells);
    void _process_predictive_coding();
    void _update_cross_modal_associations();
    void _store_episodic_memory();
    void _recall_episodic_memory(const std::vector<float>& cue);
    void _apply_recalled_memory();

    void _step_swr_replay();
    void _process_brain_state();
    std::vector<float> _get_ca3_activity() const;
    std::vector<float> _get_ca1_activity() const;
    std::vector<float> _get_cortical_activity() const;
    std::vector<std::vector<float>> _build_weight_matrix(size_t pop_idx,
        uint32_t local_start, uint32_t local_count) const;
    std::vector<std::vector<float>> _build_inter_region_weights(
        size_t src_idx, uint32_t src_start, uint32_t src_count,
        size_t dst_idx, uint32_t dst_start, uint32_t dst_count) const;
    void _extract_hippocampal_weights(
        std::vector<std::vector<float>>& dg_w,
        std::vector<std::vector<float>>& ca3_w,
        std::vector<std::vector<float>>& ca1_w) const;
};