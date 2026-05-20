#include "active_inference_engine.h"
#include <sstream>

ActiveInferenceEngine::ActiveInferenceEngine()
    : n_plans(0), active_plan_idx(-1),
      prior_preferences(STATE_DIM, 0.1f),
      avg_free_energy(0.5f), planning_depth(0.3f),
      action_confidence(0.5f), exploration_bonus(0.2f),
      step_counter(0)
{
    current_belief.hidden_state.assign(BELIEF_DIM, 0.1f);
    current_belief.precision.assign(BELIEF_DIM, 0.5f);
    current_belief.free_energy = 1.0f;
    current_belief.model_evidence = 0.5f;

    _transition_model.assign(STATE_DIM * ACTION_DIM * STATE_DIM, 0.0f);

    for (int i = 0; i < MAX_PLANS; i++) {
        action_plans[i].action_state.assign(ACTION_DIM, 0.0f);
        action_plans[i].expected_outcome.assign(STATE_DIM, 0.0f);
        action_plans[i].expected_free_energy = 10.0f;
        action_plans[i].expected_value = 0.0f;
        action_plans[i].confidence = 0.0f;
        action_plans[i].horizon = 1;
    }
}

void ActiveInferenceEngine::init(uint32_t seed) {
    n_plans = 0;
    active_plan_idx = -1;
    prior_preferences.assign(STATE_DIM, 0.1f);
    avg_free_energy = 0.5f;
    planning_depth = 0.3f;
    action_confidence = 0.5f;
    exploration_bonus = 0.2f;
    step_counter = 0;
    expected_free_energy_history.clear();
    current_belief.hidden_state.assign(BELIEF_DIM, 0.1f);
    current_belief.precision.assign(BELIEF_DIM, 0.5f);
    current_belief.free_energy = 1.0f;
    current_belief.model_evidence = 0.5f;
    for (int i = 0; i < MAX_PLANS; i++) {
        action_plans[i].expected_free_energy = 10.0f;
        action_plans[i].expected_value = 0.0f;
    }
}

void ActiveInferenceEngine::update_beliefs(
    const std::vector<float>& observation,
    const std::vector<float>& action_taken,
    float prediction_error)
{
    step_counter++;

    for (int d = 0; d < BELIEF_DIM && d < (int)observation.size(); d++) {
        float obs = observation[d];
        float pred = current_belief.hidden_state[d];
        float error = obs - pred;

        current_belief.hidden_state[d] += error * current_belief.precision[d] * 0.2f;
        current_belief.hidden_state[d] = std::max(0.0f, std::min(1.0f,
            current_belief.hidden_state[d]));
    }

    float total_err = 0.0f;
    for (int d = 0; d < BELIEF_DIM; d++) {
        float diff = d < (int)observation.size()
            ? observation[d] - current_belief.hidden_state[d] : 0.0f;
        total_err += diff * diff;
    }
    total_err = std::sqrt(total_err / (float)BELIEF_DIM);

    for (int d = 0; d < BELIEF_DIM; d++) {
        current_belief.precision[d] = current_belief.precision[d] * 0.9f
            + (1.0f - total_err) * 0.1f;
        current_belief.precision[d] = std::max(0.1f, std::min(0.9f,
            current_belief.precision[d]));
    }

    current_belief.free_energy = total_err;
    current_belief.model_evidence = 1.0f - total_err;

    if (!action_taken.empty()) {
        _update_transition_model(
            current_belief.hidden_state,
            action_taken,
            observation,
            0.01f);
    }
}

void ActiveInferenceEngine::generate_plans(
    const std::vector<float>& current_state,
    const std::vector<float>& goal_state,
    const std::vector<float>& world_model_prediction,
    const std::vector<float>& emotion_vector,
    float dopamine,
    float curiosity)
{
    n_plans = 0;

    for (int p = 0; p < MAX_PLANS; p++) {
        if (n_plans >= MAX_PLANS) break;

        auto& plan = action_plans[p];

        int horizon = 1 + (p % 3);
        plan.horizon = horizon;

        float seed = (float)((p * 1103515245 + (int)step_counter * 25214903917ULL) % 10000) / 10000.0f;

        plan.action_state.assign(ACTION_DIM, 0.0f);
        for (int d = 0; d < ACTION_DIM; d++) {
            float sval = d < (int)current_state.size() ? current_state[d] : 0.0f;
            float gval = d < (int)goal_state.size() ? goal_state[d] : 0.0f;
            float wval = d < (int)world_model_prediction.size() ? world_model_prediction[d] : 0.0f;

            float noise = (float)(((d * 7 + p * 13) * 1103515245) % 2000 - 1000) / 1000.0f * 0.2f;

            float emotion_bias = 0.0f;
            if (d < (int)emotion_vector.size()) {
                emotion_bias = emotion_vector[d % emotion_vector.size()] * 0.15f;
            }

            float direction = gval - sval;
            float exploration = curiosity * noise;

            plan.action_state[d] = direction * 0.5f + wval * 0.2f
                + exploration * 0.2f + emotion_bias;
            plan.action_state[d] = std::max(-1.0f, std::min(1.0f, plan.action_state[d]));
        }

        plan.expected_outcome.assign(STATE_DIM, 0.0f);
        for (int d = 0; d < STATE_DIM; d++) {
            float sval = d < (int)current_state.size() ? current_state[d] : 0.0f;
            float aval = d < ACTION_DIM ? plan.action_state[d] : 0.0f;
            float wval = d < (int)world_model_prediction.size() ? world_model_prediction[d] : 0.0f;

            plan.expected_outcome[d] = sval + aval * 0.3f + wval * 0.1f;
            plan.expected_outcome[d] = std::max(0.0f, std::min(1.0f,
                plan.expected_outcome[d]));
        }

        plan.expected_value = _cosine_sim(plan.expected_outcome, goal_state);
        plan.confidence = 0.3f + dopamine * 0.3f
            + (1.0f - seed * 0.3f) * 0.4f;

        plan.expected_free_energy = compute_expected_free_energy(
            plan, prior_preferences);

        n_plans++;
    }
}

void ActiveInferenceEngine::evaluate_plans(
    const std::vector<float>& current_state,
    const std::vector<float>& goal_direction,
    float risk_aversion,
    float temporal_discount)
{
    for (int p = 0; p < n_plans; p++) {
        auto& plan = action_plans[p];

        float goal_alignment = _cosine_sim(
            plan.expected_outcome, goal_direction);
        goal_alignment = std::max(0.0f, goal_alignment);

        float goal_dist = 0.0f;
        int count = 0;
        for (int d = 0; d < STATE_DIM && d < (int)goal_direction.size(); d++) {
            float diff = plan.expected_outcome[d] - goal_direction[d];
            goal_dist += diff * diff;
            count++;
        }
        float goal_error = count > 0 ? std::sqrt(goal_dist / (float)count) : 1.0f;

        float horizon_discount = std::pow(temporal_discount, (float)plan.horizon);

        plan.expected_free_energy = goal_error * 0.5f
            - goal_alignment * 0.3f
            + risk_aversion * std::abs(plan.action_state[0]) * 0.1f;

        plan.expected_value = plan.expected_value * 0.25f
            + goal_alignment * horizon_discount * 0.5f
            + (1.0f - goal_error) * horizon_discount * 0.25f;

        plan.expected_value = std::max(0.0f, plan.expected_value);
    }

    if (n_plans > 0) {
        float total_ef = 0.0f;
        for (int p = 0; p < n_plans; p++) {
            total_ef += action_plans[p].expected_free_energy;
        }
        avg_free_energy = avg_free_energy * 0.9f
            + (total_ef / (float)n_plans) * 0.1f;

        expected_free_energy_history.push_back(avg_free_energy);
        if ((int)expected_free_energy_history.size() > 50)
            expected_free_energy_history.erase(expected_free_energy_history.begin());
    }
}

void ActiveInferenceEngine::select_plan(
    float meta_confidence, float randomness)
{
    active_plan_idx = -1;

    if (n_plans == 0) return;

    float best_score = -1e10f;
    int greedy_choice = -1;

    for (int p = 0; p < n_plans; p++) {
        float score = action_plans[p].expected_value * (0.5f + meta_confidence * 0.5f)
            - action_plans[p].expected_free_energy * (1.0f - meta_confidence) * 0.3f;
        if (score > best_score) {
            best_score = score;
            greedy_choice = p;
        }
    }

    float explore_prob = randomness * 0.3f + exploration_bonus * 0.2f;
    float roll = (float)((step_counter * 1103515245) % 10000) / 10000.0f;

    if (n_plans > 1 && roll < explore_prob) {
        active_plan_idx = (int)(((uint64_t)(step_counter * 25214903917ULL) % n_plans));
    } else {
        active_plan_idx = greedy_choice;
    }

    action_confidence = (active_plan_idx >= 0 && active_plan_idx < n_plans)
        ? action_plans[active_plan_idx].confidence
        : 0.0f;

    planning_depth = planning_depth * 0.95f
        + (active_plan_idx >= 0 ? (float)action_plans[active_plan_idx].horizon / (float)MAX_HORIZON : 0.0f) * 0.05f;
}

std::vector<float> ActiveInferenceEngine::get_planned_action() const {
    std::vector<float> action(ACTION_DIM, 0.0f);
    if (active_plan_idx < 0 || active_plan_idx >= n_plans)
        return action;

    const auto& plan = action_plans[active_plan_idx];
    for (int d = 0; d < ACTION_DIM; d++) {
        action[d] = plan.action_state[d] * (0.3f + plan.confidence * 0.7f);
        action[d] = std::max(-1.0f, std::min(1.0f, action[d]));
    }
    return action;
}

std::vector<float> ActiveInferenceEngine::get_predicted_outcome() const {
    std::vector<float> outcome(STATE_DIM, 0.0f);
    if (active_plan_idx < 0 || active_plan_idx >= n_plans)
        return outcome;

    return action_plans[active_plan_idx].expected_outcome;
}

std::string ActiveInferenceEngine::get_plan_description() const {
    std::ostringstream oss;
    if (active_plan_idx < 0 || active_plan_idx >= n_plans)
        return "no_plan";

    const auto& plan = action_plans[active_plan_idx];
    oss << "h" << plan.horizon << "_";

    if (plan.expected_value > 0.7f) oss << "confident";
    else if (plan.expected_value > 0.4f) oss << "moderate";
    else oss << "uncertain";

    oss << "_c" << (int)(plan.confidence * 100);

    return oss.str();
}

float ActiveInferenceEngine::compute_expected_free_energy(
    const PlannedAction& plan,
    const std::vector<float>& goal) const
{
    float epistemic = 0.0f;
    for (int d = 0; d < STATE_DIM && d < (int)plan.expected_outcome.size(); d++) {
        float pred = plan.expected_outcome[d];
        epistemic += -pred * std::log(std::max(0.01f, pred));
        if (std::isfinite(epistemic)) {
            epistemic = epistemic * 0.5f;
        }
    }

    float pragmatic = 0.0f;
    int count = 0;
    for (int d = 0; d < STATE_DIM && d < (int)goal.size(); d++) {
        float diff = plan.expected_outcome[d] - goal[d];
        pragmatic += diff * diff;
        count++;
    }
    if (count > 0) pragmatic = std::sqrt(pragmatic / (float)count);

    return epistemic * 0.3f + pragmatic * 0.7f;
}

std::vector<float> ActiveInferenceEngine::get_policy_vector() const {
    std::vector<float> policy(ACTION_DIM, 0.0f);
    if (active_plan_idx < 0 || active_plan_idx >= n_plans)
        return policy;

    const auto& plan = action_plans[active_plan_idx];
    for (int d = 0; d < ACTION_DIM; d++) {
        policy[d] = plan.action_state[d] * plan.expected_value;
    }
    return policy;
}

void ActiveInferenceEngine::_update_transition_model(
    const std::vector<float>& prev_state,
    const std::vector<float>& action,
    const std::vector<float>& next_state,
    float lr)
{
    for (int s = 0; s < STATE_DIM && s < (int)prev_state.size(); s++) {
        for (int a = 0; a < ACTION_DIM && a < (int)action.size(); a++) {
            for (int ns = 0; ns < STATE_DIM && ns < (int)next_state.size(); ns++) {
                size_t idx = (size_t)s * ACTION_DIM * STATE_DIM + (size_t)a * STATE_DIM + (size_t)ns;
                if (idx < _transition_model.size()) {
                    float pred = prev_state[s] * action[a];
                    float target = next_state[ns];
                    _transition_model[idx] += lr * (target - pred) * prev_state[s] * action[a];
                    _transition_model[idx] = std::max(-0.5f, std::min(0.5f,
                        _transition_model[idx]));
                }
            }
        }
    }
}

float ActiveInferenceEngine::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        if (!is_bad(a[i]) && !is_bad(b[i])) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
    }
    float denom = std::sqrt(std::max(1e-10f, na * nb));
    float sim = dot / denom;
    return std::max(-1.0f, std::min(1.0f, sim));
}