#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include <deque>

struct GWBroadcastEvent {
    uint32_t target_neuron_id;
    float strength;
    uint32_t delay_ms;
};

struct GWWinnerInfo {
    int32_t winner_id;
    float activity;
    float margin_over_runner_up;
    uint32_t consecutive_wins;
    float fatigue;
};

class GlobalWorkspacePopulation {
public:
    GlobalWorkspacePopulation(uint32_t num_neurons, uint32_t competition_window_ms);

    int32_t step(const std::vector<std::pair<uint32_t, float>>& inputs,
                  float dopamine);
    std::vector<GWBroadcastEvent> broadcast(uint32_t target_population_size);

    bool is_ignited() const { return ignition_strength >= 0.55f && competition_entropy >= 0.15f; }
    float get_ignition_strength() const { return ignition_strength; }
    float get_phi_estimate() const;
    float get_activity_level() const { return global_activity; }
    float get_competition_entropy() const { return competition_entropy; }
    float get_winner_margin() const { return winner_margin; }
    float get_fatigue_level() const { return winner_fatigue; }
    float get_ignition_saturation_risk() const { return ignition_saturation_risk; }
    float get_inhibition_pressure() const { return adaptive_inhibition_scale; }
    uint32_t get_winner_diversity() const { return unique_winners_count; }

    uint32_t get_winner() const { return current_winner; }
    uint32_t size() const { return num_neurons; }
    uint32_t get_ignition_count() const { return ignition_count; }

    GWWinnerInfo get_winner_info() const;

    void reset();
    void set_seed(uint64_t seed) { rng.seed(seed); }

private:
    uint32_t num_neurons;
    uint32_t competition_window_ms;
    uint64_t step_counter;

    std::vector<float> membrane;
    std::vector<float> recovery;
    std::vector<float> excitation;
    std::vector<float> inhibition;
    std::vector<float> activity;

    int32_t current_winner;
    int32_t previous_winner;
    uint32_t ignition_count;
    uint32_t winner_consecutive_count;
    float global_activity;
    float prev_activity;
    float ignition_strength;
    float competition_entropy;
    float winner_margin;
    float winner_fatigue;
    float adaptive_inhibition_scale;
    float ignition_saturation_risk;
    uint32_t unique_winners_count;

    std::deque<float> activity_history;
    std::deque<float> entropy_history;
    std::deque<int32_t> winner_history;
    static constexpr size_t MAX_HISTORY = 100;

    std::mt19937 rng;

    void _update_izhikevich(float dopamine);
    void _apply_lateral_inhibition();
    int32_t _select_winner();
    float _compute_phi();
    void _update_fatigue();
    void _update_ignition_strength();
    float _compute_activity_entropy();
    void _apply_winner_refractory();
};