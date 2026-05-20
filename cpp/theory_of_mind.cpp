#include "theory_of_mind.h"
#include <random>
#include <sstream>

TheoryOfMind::TheoryOfMind()
    : n_agents(0), self_other_boundary(64, 0.5f),
      self_other_separation(0.3f), false_belief_accuracy(0.0f),
      empathy_level(0.1f), social_awareness(0.1f),
      step_counter(0)
{
    social_ctx.group_norm.assign(SOCIAL_CONTEXT_DIM, 0.0f);
    social_ctx.social_hierarchy.assign(SOCIAL_CONTEXT_DIM, 0.0f);
    social_ctx.cooperation_signal = 0.0f;
    social_ctx.competition_signal = 0.0f;
    social_ctx.social_anxiety = 0.0f;
    for (int i = 0; i < MAX_AGENTS; i++) {
        agents[i].belief_state.assign(BELIEF_DIM, 0.0f);
        agents[i].desire_state.assign(DESIRE_DIM, 0.0f);
        agents[i].intention_state.assign(INTENTION_DIM, 0.0f);
        agents[i].perceived_location.assign(4, 0.0f);
        agents[i].recent_actions.assign(8, 0.0f);
        agents[i].familiarity = 0.0f;
        agents[i].trust_level = 0.5f;
        agents[i].perceived_competence = 0.3f;
        agents[i].last_interaction = 0;
        agents[i].identity_id = i;
        agents[i].is_present = false;
    }
}

void TheoryOfMind::init(uint32_t seed) {
    n_agents = 0;
    self_other_boundary.assign(64, 0.5f);
    self_other_separation = 0.3f;
    false_belief_accuracy = 0.0f;
    empathy_level = 0.1f;
    social_awareness = 0.1f;
    step_counter = 0;
    social_ctx.cooperation_signal = 0.0f;
    social_ctx.competition_signal = 0.0f;
    social_ctx.social_anxiety = 0.0f;
    for (int i = 0; i < MAX_AGENTS; i++) {
        agents[i].belief_state.assign(BELIEF_DIM, 0.0f);
        agents[i].desire_state.assign(DESIRE_DIM, 0.0f);
        agents[i].intention_state.assign(INTENTION_DIM, 0.0f);
        agents[i].familiarity = 0.0f;
        agents[i].trust_level = 0.5f;
        agents[i].is_present = false;
    }
}

void TheoryOfMind::perceive_agent(
    int agent_id,
    const std::vector<float>& observed_state,
    const std::vector<float>& observed_action,
    float interaction_outcome)
{
    if (agent_id < 0 || agent_id >= MAX_AGENTS) return;

    auto& ag = agents[agent_id];
    if (!ag.is_present) {
        ag.is_present = true;
        if (agent_id >= n_agents) n_agents = agent_id + 1;
    }

    for (int d = 0; d < BELIEF_DIM && d < (int)observed_state.size(); d++)
        ag.belief_state[d] = ag.belief_state[d] * 0.7f + observed_state[d] * 0.3f;

    for (int d = 0; d < 8 && d < (int)observed_action.size(); d++)
        ag.recent_actions[d] = observed_action[d];

    ag.familiarity = std::min(1.0f, ag.familiarity + 0.02f);
    ag.last_interaction = step_counter;

    float outcome_effect = interaction_outcome * 0.1f;
    ag.trust_level = std::max(0.0f, std::min(1.0f,
        ag.trust_level + outcome_effect));

    if (std::abs(interaction_outcome) > 0.3f) {
        ag.perceived_competence = ag.perceived_competence * 0.9f
            + std::abs(interaction_outcome) * 0.1f;
    }
}

void TheoryOfMind::update_beliefs(
    int agent_id,
    const std::vector<float>& world_state,
    const std::vector<float>& self_state)
{
    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present)
        return;

    auto& ag = agents[agent_id];

    for (int d = 0; d < BELIEF_DIM; d++) {
        float wval = d < (int)world_state.size() ? world_state[d] : 0.0f;
        ag.belief_state[d] = ag.belief_state[d] * 0.85f + wval * 0.15f;
    }

    for (int d = 0; d < DESIRE_DIM; d++) {
        float sval = d < (int)self_state.size() ? self_state[d] : 0.0f;
        ag.desire_state[d] = ag.desire_state[d] * 0.9f + sval * 0.1f;
    }
}

void TheoryOfMind::infer_intentions(
    int agent_id,
    const std::vector<float>& recent_history)
{
    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present)
        return;

    auto& ag = agents[agent_id];

    for (int d = 0; d < INTENTION_DIM; d++) {
        float hval = d < (int)recent_history.size() ? recent_history[d] : 0.0f;
        float bval = d < BELIEF_DIM ? ag.belief_state[d] : 0.0f;
        float dval = d < DESIRE_DIM ? ag.desire_state[d] : 0.0f;
        ag.intention_state[d] = bval * 0.4f + dval * 0.4f + hval * 0.2f;
    }
}

SocialInference TheoryOfMind::predict_other(
    int agent_id,
    const std::vector<float>& current_world,
    const std::vector<float>& self_intention)
{
    SocialInference inf;
    inf.agent_id = agent_id;
    inf.empathy_signal = 0.0f;
    inf.false_belief_detected = false;

    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present) {
        inf.predicted_action.assign(8, 0.0f);
        inf.prediction_confidence = 0.1f;
        inf.perspective_taking.assign(16, 0.0f);
        return inf;
    }

    auto& ag = agents[agent_id];
    inf.predicted_action.assign(8, 0.0f);

    for (int d = 0; d < 8; d++) {
        float intent = d < INTENTION_DIM ? ag.intention_state[d] : 0.0f;
        float world = d < (int)current_world.size() ? current_world[d] : 0.0f;
        inf.predicted_action[d] = intent * 0.6f + world * 0.4f;
        inf.predicted_action[d] = std::max(0.0f, std::min(1.0f, inf.predicted_action[d]));
    }

    inf.prediction_confidence = 0.3f + ag.familiarity * 0.4f + ag.perceived_competence * 0.3f;

    inf.perspective_taking.assign(16, 0.0f);
    for (int d = 0; d < 16; d++) {
        float belief = d < BELIEF_DIM ? ag.belief_state[d] : 0.0f;
        float desire = d < DESIRE_DIM ? ag.desire_state[d] : 0.0f;
        inf.perspective_taking[d] = belief * 0.5f + desire * 0.5f;
    }

    float sim = 0.0f;
    for (int d = 0; d < 16 && d < (int)self_intention.size(); d++) {
        sim += inf.perspective_taking[d] * self_intention[d];
    }
    sim = std::max(-1.0f, std::min(1.0f, sim * 2.0f));
    inf.empathy_signal = ag.familiarity * 0.5f + sim * 0.5f;

    return inf;
}

bool TheoryOfMind::detect_false_belief(
    int agent_id,
    const std::vector<float>& ground_truth)
{
    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present)
        return false;

    auto& ag = agents[agent_id];
    float diff = 0.0f;
    int count = 0;
    for (int d = 0; d < BELIEF_DIM && d < (int)ground_truth.size(); d++) {
        float delta = ag.belief_state[d] - ground_truth[d];
        diff += delta * delta;
        count++;
    }
    if (count == 0) return false;
    float rmse = std::sqrt(diff / (float)count);

    bool detected = rmse > 0.3f;
    if (detected) {
        false_belief_accuracy = false_belief_accuracy * 0.9f + 1.0f * 0.1f;
    } else {
        false_belief_accuracy = false_belief_accuracy * 0.9f;
    }
    return detected;
}

std::vector<float> TheoryOfMind::take_perspective(
    int agent_id,
    const std::vector<float>& situation) const
{
    std::vector<float> perspective(16, 0.0f);
    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present)
        return perspective;

    const auto& ag = agents[agent_id];
    for (int d = 0; d < 16; d++) {
        float belief = d < BELIEF_DIM ? ag.belief_state[d] : 0.0f;
        float sit = d < (int)situation.size() ? situation[d] : 0.0f;
        perspective[d] = belief * 0.6f + sit * 0.4f;
    }
    return perspective;
}

void TheoryOfMind::update_self_other_boundary(
    const std::vector<float>& self_state,
    const std::vector<float>& other_observation)
{
    for (int d = 0; d < 64 && d < (int)self_state.size(); d++) {
        float sval = self_state[d];
        float oval = d < (int)other_observation.size() ? other_observation[d] : 0.0f;
        float diff = std::abs(sval - oval);
        self_other_boundary[d] = self_other_boundary[d] * 0.95f + diff * 0.05f;
    }

    float separation = 0.0f;
    for (auto v : self_other_boundary) separation += v;
    separation /= (float)self_other_boundary.size();
    self_other_separation = self_other_separation * 0.9f + separation * 0.1f;
}

void TheoryOfMind::update_social_context(
    const std::vector<float>& interaction_valence,
    float cooperation_level,
    float reward_shared)
{
    social_ctx.cooperation_signal = social_ctx.cooperation_signal * 0.8f
        + cooperation_level * 0.2f;
    social_ctx.competition_signal = social_ctx.competition_signal * 0.8f
        + (1.0f - cooperation_level) * 0.2f;

    float anxiety = 0.0f;
    for (auto v : interaction_valence) anxiety += v;
    anxiety /= std::max(1.0f, (float)interaction_valence.size());
    social_ctx.social_anxiety = social_ctx.social_anxiety * 0.85f
        + (1.0f - std::abs(anxiety)) * 0.15f;

    social_awareness = social_awareness * 0.95f
        + (cooperation_level * 0.5f + std::abs(reward_shared) * 0.5f) * 0.05f;

    empathy_level = empathy_level * 0.9f
        + cooperation_level * 0.05f + social_awareness * 0.05f;
}

std::vector<float> TheoryOfMind::get_social_modulation() const {
    std::vector<float> mod(32, 0.0f);
    for (int d = 0; d < 32; d++) {
        float coop = d < SOCIAL_CONTEXT_DIM ? social_ctx.group_norm[d] : 0.0f;
        float comp = social_ctx.competition_signal;
        mod[d] = coop * 0.5f + comp * 0.2f + social_awareness * 0.3f;
    }
    return mod;
}

std::vector<float> TheoryOfMind::get_empathy_vector(int agent_id) const {
    if (agent_id < 0 || agent_id >= MAX_AGENTS || !agents[agent_id].is_present)
        return std::vector<float>(16, 0.0f);

    const auto& ag = agents[agent_id];
    std::vector<float> empathy(16, 0.0f);
    for (int d = 0; d < 16; d++) {
        float belief = d < BELIEF_DIM ? ag.belief_state[d] : 0.0f;
        float desire = d < DESIRE_DIM ? ag.desire_state[d] : 0.0f;
        empathy[d] = belief * 0.4f + desire * 0.4f;
        empathy[d] *= ag.familiarity;
    }
    return empathy;
}

std::string TheoryOfMind::get_social_narrative() const {
    std::ostringstream oss;
    if (n_agents == 0) return "solitary";
    oss << "social_";
    if (social_ctx.cooperation_signal > 0.5f) oss << "coop";
    else if (social_ctx.competition_signal > 0.5f) oss << "comp";
    else oss << "neutral";
    oss << "_" << n_agents << "others";
    oss << "_emp" << (int)(empathy_level * 100);
    return oss.str();
}

float TheoryOfMind::get_theory_of_mind_level() const {
    return self_other_separation * 0.3f
        + false_belief_accuracy * 0.25f
        + std::min(1.0f, (float)n_agents / 3.0f) * 0.15f
        + social_awareness * 0.15f
        + empathy_level * 0.15f;
}

float TheoryOfMind::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(std::max(1e-10f, na * nb));
    return dot / denom;
}

void TheoryOfMind::_normalize(std::vector<float>& v) const {
    float norm = 0.0f;
    for (auto x : v) if (!is_bad(x)) norm += x * x;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& x : v) x = x / norm * 0.5f;
    }
}