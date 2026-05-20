#include "action_selector.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

ActionSelector::ActionSelector()
    : history_pos(0), history_fill(0), epsilon(0.3f),
      learning_rate(0.01f), steps(0)
{
    std::random_device rd;
    rng.seed(rd());
}

void ActionSelector::init() {
    action_effects.assign(N_ACTIONS * SENSORY_DIM, 0.0f);
    action_counts.assign(N_ACTIONS, 0);
    action_rewards.assign(N_ACTIONS, 0.0f);
    recent_sensory.assign(HISTORY_LEN, std::vector<float>(SENSORY_DIM, 0.0f));
    recent_actions.assign(HISTORY_LEN, 0);
    history_pos = 0;
    history_fill = 0;
    epsilon = 0.3f;
    steps = 0;
}

std::vector<float> ActionSelector::predict_action_outcome(
    const std::vector<float>& world_prediction, int action) const
{
    if (action < 0 || action >= (int)N_ACTIONS) return world_prediction;
    size_t base = action * SENSORY_DIM;
    std::vector<float> result = world_prediction;
    for (size_t d = 0; d < SENSORY_DIM && d < result.size(); d++) {
        float effect = action_effects[base + d];
        if (!is_bad(effect)) result[d] += effect;
    }
    return result;
}

float ActionSelector::compute_efe(const std::vector<float>& predicted_next,
                                     const std::vector<float>& target,
                                     int action, float curiosity) const
{
    float pred_error = 0.0f;
    size_t n = std::min(predicted_next.size(), target.size());
    for (size_t d = 0; d < n; d++) {
        float diff = predicted_next[d] - target[d];
        if (!is_bad(diff)) pred_error += diff * diff;
    }
    pred_error /= std::max(1.0f, (float)n);

    float info_gain = 0.0f;
    if (action >= 0 && action < (int)N_ACTIONS) {
        int count = action_counts[action];
        info_gain = (count < 5) ? 1.0f : 1.0f / std::sqrt((float)count);
    }

    float avg_reward = 0.0f;
    if (action >= 0 && action < (int)N_ACTIONS && action_counts[action] > 0) {
        avg_reward = action_rewards[action] / (float)action_counts[action];
    }

    return pred_error - curiosity * info_gain * 0.3f - avg_reward * 0.2f;
}

int ActionSelector::select_action(const std::vector<float>& current_sensory,
                                    const std::vector<float>& world_prediction,
                                    float curiosity, float dopamine) {
    steps++;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) < epsilon) {
        std::uniform_int_distribution<int> adist(0, N_ACTIONS - 1);
        return adist(rng);
    }

    float best_efe = 1e10f;
    int best_action = 0;

    for (int a = 0; a < (int)N_ACTIONS; a++) {
        auto pred_next = predict_action_outcome(world_prediction, a);
        float efe = compute_efe(pred_next, current_sensory, a, curiosity);
        if (is_bad(efe)) efe = 1e10f;
        if (efe < best_efe) {
            best_efe = efe;
            best_action = a;
        }
    }

    return best_action;
}

void ActionSelector::learn(const std::vector<float>& prev_sensory,
                            const std::vector<float>& actual_next,
                            int action_taken, float reward) {
    if (action_taken < 0 || action_taken >= (int)N_ACTIONS) return;

    action_counts[action_taken]++;
    action_rewards[action_taken] += reward;
    epsilon = std::max(0.05f, epsilon * 0.9995f);

    size_t base = action_taken * SENSORY_DIM;
    float lr = learning_rate / std::sqrt((float)std::max(1, action_counts[action_taken]));

    for (size_t d = 0; d < SENSORY_DIM && d < prev_sensory.size() && d < actual_next.size(); d++) {
        float target_effect = actual_next[d] - prev_sensory[d];
        float current = action_effects[base + d];
        if (is_bad(target_effect)) target_effect = 0.0f;
        if (is_bad(current)) current = 0.0f;
        action_effects[base + d] = current * 0.95f + target_effect * lr;
        if (action_effects[base + d] > 0.3f) action_effects[base + d] = 0.3f;
        if (action_effects[base + d] < -0.3f) action_effects[base + d] = -0.3f;
    }

    if (history_fill < HISTORY_LEN) {
        recent_sensory[history_fill] = prev_sensory;
        recent_actions[history_fill] = action_taken;
        history_fill++;
    } else {
        recent_sensory[history_pos] = prev_sensory;
        recent_actions[history_pos] = action_taken;
        history_pos = (history_pos + 1) % HISTORY_LEN;
    }
}