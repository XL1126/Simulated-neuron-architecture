#include "global_workspace.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

GlobalWorkspacePopulation::GlobalWorkspacePopulation(
    uint32_t num_neurons_, uint32_t competition_window_ms_)
    : num_neurons(num_neurons_), competition_window_ms(competition_window_ms_),
      step_counter(0),
      current_winner(-1), previous_winner(-1),
      ignition_count(0), winner_consecutive_count(0),
      global_activity(0.0f), prev_activity(0.0f),
      ignition_strength(0.0f), competition_entropy(0.0f),
      winner_margin(0.0f), winner_fatigue(0.0f),
      adaptive_inhibition_scale(0.35f),
      ignition_saturation_risk(0.0f),
      unique_winners_count(0)
{
    std::random_device rd;
    rng.seed(rd());

    membrane.resize(num_neurons, -65.0f);
    recovery.resize(num_neurons, 0.0f);
    excitation.resize(num_neurons, 0.0f);
    inhibition.resize(num_neurons, 0.0f);
    activity.resize(num_neurons, 0.01f);

    std::uniform_real_distribution<float> init_dist(-5.0f, 5.0f);
    for (uint32_t i = 0; i < num_neurons; i++) {
        membrane[i] = -65.0f + init_dist(rng);
        recovery[i] = membrane[i] * 0.2f;
    }
}

float GlobalWorkspacePopulation::_compute_activity_entropy() {
    float total = 0.0f;
    for (auto a : activity) total += a;
    if (total < 1e-8f) return 0.0f;

    float entropy = 0.0f;
    for (auto a : activity) {
        float p = a / total;
        if (p > 1e-8f) entropy -= p * std::log(p);
    }
    float max_entropy = std::log((float)num_neurons);
    if (max_entropy < 1e-8f) return 0.0f;
    return entropy / max_entropy;
}

void GlobalWorkspacePopulation::_update_izhikevich(float dopamine) {
    for (uint32_t i = 0; i < num_neurons; i++) {
        float I_total = excitation[i] - inhibition[i]
            + std::normal_distribution<float>(0.0f, 0.5f)(rng);
        excitation[i] *= 0.85f;

        float v = membrane[i];
        float u = recovery[i];
        float dv = 0.04f * v * v + 5.0f * v + 140.0f - u + I_total;
        float du = 0.02f * (0.2f * v - u);
        membrane[i] += dv * 0.5f;
        recovery[i] += du * 0.5f;

        if (membrane[i] >= 30.0f) {
            membrane[i] = -65.0f;
            recovery[i] += 2.0f;
            activity[i] = std::min(1.0f, activity[i] + 0.2f);
        }

        activity[i] *= 0.92f;
        inhibition[i] *= 0.85f;
        membrane[i] += dopamine * 0.3f;
    }
}

void GlobalWorkspacePopulation::_apply_lateral_inhibition() {
    float total = 0.0f;
    for (auto a : activity) total += a;
    if (total < 0.005f) return;

    float mean_activity = total / (float)num_neurons;

    for (uint32_t i = 0; i < num_neurons; i++) {
        float inhibition_strength = (total - activity[i]) * adaptive_inhibition_scale;
        inhibition[i] += inhibition_strength * 2.0f;
    }

    float entropy = _compute_activity_entropy();
    competition_entropy = entropy;

    if (entropy > 0.7f) {
        adaptive_inhibition_scale = std::min(0.65f, adaptive_inhibition_scale + 0.005f);
    } else if (entropy < 0.25f) {
        adaptive_inhibition_scale = std::max(0.15f, adaptive_inhibition_scale - 0.008f);
    } else {
        adaptive_inhibition_scale = adaptive_inhibition_scale * 0.995f + 0.30f * 0.005f;
    }

    float stable_wins = 0.0f;
    if (winner_history.size() >= 20) {
        size_t start = winner_history.size() - 20;
        for (size_t i = start + 1; i < winner_history.size(); i++) {
            if (winner_history[i] == winner_history[start]) stable_wins += 1.0f;
        }
    }
    ignition_saturation_risk = std::min(1.0f, stable_wins / 15.0f);
    if (ignition_saturation_risk > 0.6f) {
        adaptive_inhibition_scale = std::min(0.65f,
            adaptive_inhibition_scale + 0.012f * ignition_saturation_risk);
    }

    activity_history.push_back(global_activity);
    if (activity_history.size() > MAX_HISTORY) activity_history.pop_front();
    entropy_history.push_back(entropy);
    if (entropy_history.size() > MAX_HISTORY) entropy_history.pop_front();
}

void GlobalWorkspacePopulation::_update_fatigue() {
    if (current_winner >= 0 && current_winner == previous_winner) {
        winner_consecutive_count++;
        winner_fatigue = std::min(1.0f,
            0.15f * (float)winner_consecutive_count
            + 0.08f * activity[current_winner]);
    } else {
        winner_consecutive_count = 0;
        winner_fatigue *= 0.55f;
    }
    winner_fatigue = std::max(0.0f, winner_fatigue * 0.96f);
}

void GlobalWorkspacePopulation::_apply_winner_refractory() {
    if (winner_fatigue > 0.5f && current_winner >= 0) {
        uint32_t widx = (uint32_t)current_winner;
        activity[widx] *= 0.35f;
        membrane[widx] = -70.0f;
        inhibition[widx] += 8.0f * winner_fatigue;
    }
}

void GlobalWorkspacePopulation::_update_ignition_strength() {
    float base = 0.0f;

    if (current_winner >= 0) {
        base = std::min(1.0f, activity[current_winner] * 2.5f);
        base *= (1.0f - winner_fatigue * 0.7f);
    }

    if (winner_margin > 0.4f) {
        base += winner_margin * 0.3f;
    } else if (winner_margin < 0.2f) {
        base *= 0.6f;
    }

    float entropy_weight = 0.4f + 0.6f * competition_entropy;
    base *= entropy_weight;

    if (entropy_history.size() >= 10) {
        float recent_entropy = 0.0f;
        size_t start = entropy_history.size() > 10 ? entropy_history.size() - 10 : 0;
        for (size_t i = start; i < entropy_history.size(); i++)
            recent_entropy += entropy_history[i];
        recent_entropy /= (float)(entropy_history.size() - start);
        if (recent_entropy < 0.2f) {
            base *= 0.25f;
        } else if (recent_entropy < 0.35f) {
            base *= 0.5f;
        }
    }

    if (ignition_saturation_risk > 0.5f) {
        base *= (1.0f - ignition_saturation_risk * 0.5f);
    }

    float alpha = 0.12f;
    ignition_strength = ignition_strength * (1.0f - alpha) + base * alpha;
    ignition_strength = std::max(0.0f, std::min(1.0f, ignition_strength));
}

int32_t GlobalWorkspacePopulation::_select_winner() {
    float max_act = 0.0f;
    int32_t winner = -1;
    float runner_up = 0.0f;

    for (uint32_t i = 0; i < num_neurons; i++) {
        if (activity[i] > max_act) {
            runner_up = max_act;
            max_act = activity[i];
            winner = (int32_t)i;
        } else if (activity[i] > runner_up) {
            runner_up = activity[i];
        }
    }

    if (winner >= 0) {
        winner_margin = (max_act > 0.0f) ? (max_act - runner_up) / max_act : 0.0f;

        float margin_threshold = 0.30f + 0.12f * winner_fatigue;
        float min_activity = 0.08f + 0.06f * winner_fatigue;

        if (max_act > runner_up * 2.5f && max_act > min_activity) {
            return winner;
        }
    }

    winner_margin = 0.0f;
    return -1;
}

float GlobalWorkspacePopulation::_compute_phi() {
    float diff = std::abs(global_activity - prev_activity);
    float integration = global_activity;

    float differentiation = 0.0f;
    for (uint32_t i = 0; i < num_neurons; i++) {
        for (uint32_t j = i + 1; j < num_neurons; j++) {
            differentiation += std::abs(activity[i] - activity[j]);
        }
    }
    uint32_t pairs = num_neurons * (num_neurons - 1) / 2;
    if (pairs > 0) differentiation /= (float)pairs;

    float phi = 0.3f * diff + 0.4f * integration + 0.3f * differentiation;

    if (competition_entropy < 0.15f) {
        phi *= (0.4f + 0.6f * (competition_entropy / 0.15f));
    }

    prev_activity = global_activity;
    return std::max(0.0f, std::min(1.0f, phi));
}

int32_t GlobalWorkspacePopulation::step(
    const std::vector<std::pair<uint32_t, float>>& inputs,
    float dopamine)
{
    step_counter++;

    for (auto& [nid, strength] : inputs) {
        uint32_t idx = nid % num_neurons;
        excitation[idx] += strength * 0.6f;
        activity[idx] = std::min(1.0f, activity[idx] + strength * 0.15f);
    }

    _update_izhikevich(dopamine);

    if (step_counter % competition_window_ms == 0) {
        _apply_lateral_inhibition();
        _update_fatigue();
        int32_t new_winner = _select_winner();

        previous_winner = current_winner;

        if (new_winner >= 0 && new_winner == current_winner) {
            ignition_count++;
        } else {
            ignition_count = (new_winner >= 0) ? 1 : 0;
            current_winner = new_winner;
        }

        if (current_winner >= 0) {
            winner_history.push_back(current_winner);
            if (winner_history.size() > MAX_HISTORY) winner_history.pop_front();

            std::set<int32_t> unique_set;
            for (auto w : winner_history) {
                if (w >= 0) unique_set.insert(w);
            }
            unique_winners_count = (uint32_t)unique_set.size();
        }

        _apply_winner_refractory();
        _update_ignition_strength();
    }

    global_activity = 0.0f;
    for (auto a : activity) global_activity += a;
    global_activity /= (float)num_neurons;

    return current_winner;
}

std::vector<GWBroadcastEvent> GlobalWorkspacePopulation::broadcast(
    uint32_t target_population_size)
{
    std::vector<GWBroadcastEvent> out_broadcasts;
    if (!is_ignited() || current_winner < 0) return out_broadcasts;

    float broadcast_strength = ignition_strength * 1.2f;
    broadcast_strength = std::max(0.1f, std::min(1.0f, broadcast_strength));

    uint32_t num_targets = (uint32_t)(std::min(target_population_size, (uint32_t)50)
        * ignition_strength);
    num_targets = std::max((uint32_t)5, std::min((uint32_t)50, num_targets));

    std::uniform_int_distribution<uint32_t> target_dist(0, target_population_size - 1);

    for (uint32_t i = 0; i < num_targets; i++) {
        GWBroadcastEvent evt;
        evt.target_neuron_id = (current_winner * 100 + i * 37 + target_dist(rng))
                               % target_population_size;
        evt.strength = broadcast_strength * (1.0f - (float)i / (float)num_targets);
        evt.delay_ms = 1;
        out_broadcasts.push_back(evt);
    }

    return out_broadcasts;
}

float GlobalWorkspacePopulation::get_phi_estimate() const {
    float diff = std::abs(global_activity - prev_activity);
    float integration = global_activity;
    float diff_total = 0.0f;
    for (uint32_t i = 0; i < num_neurons; i++) {
        for (uint32_t j = i + 1; j < num_neurons; j++) {
            diff_total += std::abs(activity[i] - activity[j]);
        }
    }
    uint32_t pairs = num_neurons * (num_neurons - 1) / 2;
    float differentiation = (pairs > 0) ? diff_total / (float)pairs : 0.0f;

    float phi = 0.3f * diff + 0.4f * integration + 0.3f * differentiation;

    if (competition_entropy < 0.15f) {
        phi *= (0.4f + 0.6f * (competition_entropy / 0.15f));
    }

    return std::max(0.0f, std::min(1.0f, phi));
}

GWWinnerInfo GlobalWorkspacePopulation::get_winner_info() const {
    GWWinnerInfo info;
    info.winner_id = current_winner;
    info.activity = (current_winner >= 0) ? activity[current_winner] : 0.0f;
    info.margin_over_runner_up = winner_margin;
    info.consecutive_wins = winner_consecutive_count;
    info.fatigue = winner_fatigue;
    return info;
}

void GlobalWorkspacePopulation::reset() {
    current_winner = -1;
    previous_winner = -1;
    ignition_count = 0;
    winner_consecutive_count = 0;
    global_activity = 0.0f;
    prev_activity = 0.0f;
    ignition_strength = 0.0f;
    competition_entropy = 0.0f;
    winner_margin = 0.0f;
    winner_fatigue = 0.0f;
    adaptive_inhibition_scale = 0.35f;
    ignition_saturation_risk = 0.0f;
    unique_winners_count = 0;

    for (uint32_t i = 0; i < num_neurons; i++) {
        activity[i] = 0.01f;
        excitation[i] = 0.0f;
        inhibition[i] = 0.0f;
        membrane[i] = -65.0f;
        recovery[i] = 0.0f;
    }

    activity_history.clear();
    entropy_history.clear();
    winner_history.clear();
}