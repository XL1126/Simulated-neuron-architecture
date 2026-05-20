#include "counterfactual_engine.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

CounterfactualEngine::CounterfactualEngine()
    : regret_ema(0.0f), counterfactual_curiosity(0.3f), steps(0)
{}

void CounterfactualEngine::init() {
    state_history.clear();
    action_history.clear();
    outcome_history.clear();
    action_value_model.assign(N_ACTIONS, 0.0f);
    transition_weights.assign(STATE_DIM * N_ACTIONS, 0.0f);
    regret_ema = 0.0f;
    counterfactual_curiosity = 0.3f;
    steps = 0;
}

void CounterfactualEngine::record_experience(
    const std::vector<float>& state, int action, float outcome) {

    std::vector<float> clipped_state(std::min(state.size(), (size_t)STATE_DIM), 0.0f);
    for (size_t i = 0; i < clipped_state.size(); i++) {
        clipped_state[i] = is_bad(state[i]) ? 0.0f : std::tanh(state[i]);
    }

    state_history.push_back(clipped_state);
    action_history.push_back(action);
    outcome_history.push_back(outcome);
    steps++;

    if ((int)state_history.size() > HISTORY_LEN) state_history.pop_front();
    if ((int)action_history.size() > HISTORY_LEN) action_history.pop_front();
    if ((int)outcome_history.size() > HISTORY_LEN) outcome_history.pop_front();

    action_value_model[action] = action_value_model[action] * 0.9f + outcome * 0.1f;
}

std::vector<CounterfactualOutcome> CounterfactualEngine::simulate_alternatives(
    const std::vector<float>& prev_state,
    int actual_action, float actual_outcome,
    const std::vector<float>& current_state,
    const std::vector<float>& world_prediction) {

    std::vector<CounterfactualOutcome> alternatives;

    size_t sdim = std::min(prev_state.size(), (size_t)STATE_DIM);

    for (int a = 0; a < N_ACTIONS; a++) {
        if (a == actual_action) continue;

        float predicted = 0.0f;
        for (size_t i = 0; i < sdim; i++) {
            float w = transition_weights[i * N_ACTIONS + a];
            float sv = is_bad(prev_state[i]) ? 0.0f : prev_state[i];
            predicted += sv * w;
        }

        predicted += action_value_model[a] * 0.3f;

        if (!world_prediction.empty()) {
            size_t wp_n = std::min(world_prediction.size(), sdim);
            for (size_t i = 0; i < wp_n; i++) {
                if (!is_bad(world_prediction[i]))
                    predicted += world_prediction[i] * 0.02f * (float)(a + 1);
            }
        }

        float regret = std::max(0.0f, predicted - actual_outcome);

        CounterfactualOutcome cf;
        cf.action = a;
        cf.predicted_outcome = predicted;
        cf.regret = regret;
        alternatives.push_back(cf);
    }

    std::sort(alternatives.begin(), alternatives.end(),
              [](const auto& a, const auto& b) { return a.regret > b.regret; });

    return alternatives;
}

void CounterfactualEngine::learn_from_regret(
    const std::vector<CounterfactualOutcome>& alternatives,
    const std::vector<float>& state) {

    float max_regret = 0.0f;
    for (auto& alt : alternatives) {
        if (alt.regret > max_regret) max_regret = alt.regret;
    }
    regret_ema = regret_ema * 0.9f + max_regret * 0.1f;

    if (max_regret > 0.1f) {
        counterfactual_curiosity = std::min(0.8f, counterfactual_curiosity + max_regret * 0.1f);
    } else {
        counterfactual_curiosity = std::max(0.15f, counterfactual_curiosity - 0.002f);
    }

    size_t sdim = std::min(state.size(), (size_t)STATE_DIM);
    for (auto& alt : alternatives) {
        if (alt.regret > 0.05f) {
            for (size_t i = 0; i < sdim; i++) {
                float sv = is_bad(state[i]) ? 0.0f : state[i];
                transition_weights[i * N_ACTIONS + alt.action] += alt.regret * 0.01f * sv;
                transition_weights[i * N_ACTIONS + alt.action] =
                    std::max(-0.5f, std::min(0.5f, transition_weights[i * N_ACTIONS + alt.action]));
            }
        }
    }
}