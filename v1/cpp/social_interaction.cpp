#include "social_interaction.h"
#include <sstream>
#include <random>

SocialInteractionModel::SocialInteractionModel()
    : n_active_agents(0),
      social_embedding(32, 0.0f),
      group_identity(IDENTITY_DIM, 0.0f),
      social_cohesion(0.3f), social_influence(0.2f),
      reciprocity(0.3f), social_learning_rate(0.05f),
      affiliation_need(0.3f), social_satisfaction(0.3f),
      step_counter(0)
{
    for (int i = 0; i < MAX_SOCIAL_AGENTS; i++) {
        social_agents[i].identity.assign(IDENTITY_DIM, 0.0f);
        social_agents[i].recent_actions.assign(8, 0.0f);
        social_agents[i].emotional_state.assign(4, 0.0f);
        social_agents[i].reputation = 0.3f;
        social_agents[i].relationship_valence = 0.0f;
        social_agents[i].reciprocity_score = 0.0f;
        social_agents[i].similarity = 0.0f;
        social_agents[i].interaction_count = 0;
        social_agents[i].last_seen = 0;
        social_agents[i].agent_id = i;
        social_agents[i].is_active = false;
    }
}

void SocialInteractionModel::init(uint32_t seed) {
    n_active_agents = 0;
    social_embedding.assign(32, 0.0f);
    group_identity.assign(IDENTITY_DIM, 0.0f);
    social_cohesion = 0.3f;
    social_influence = 0.2f;
    reciprocity = 0.3f;
    social_learning_rate = 0.05f;
    affiliation_need = 0.3f;
    social_satisfaction = 0.3f;
    step_counter = 0;
    interaction_history.clear();
    for (int i = 0; i < MAX_SOCIAL_AGENTS; i++) {
        social_agents[i].is_active = false;
        social_agents[i].interaction_count = 0;
        social_agents[i].relationship_valence = 0.0f;
    }
}

void SocialInteractionModel::add_or_update_agent(
    int agent_id,
    const std::vector<float>& observed_identity,
    const std::vector<float>& observed_action,
    const std::vector<float>& observed_emotion)
{
    if (agent_id < 0 || agent_id >= MAX_SOCIAL_AGENTS) return;

    auto& agent = social_agents[agent_id];
    if (!agent.is_active) {
        agent.is_active = true;
        if (agent_id >= n_active_agents) n_active_agents = agent_id + 1;
    }

    for (int d = 0; d < IDENTITY_DIM && d < (int)observed_identity.size(); d++) {
        agent.identity[d] = agent.identity[d] * 0.7f + observed_identity[d] * 0.3f;
    }

    for (int d = 0; d < 8 && d < (int)observed_action.size(); d++) {
        agent.recent_actions[d] = observed_action[d];
    }

    for (int d = 0; d < 4 && d < (int)observed_emotion.size(); d++) {
        agent.emotional_state[d] = observed_emotion[d];
    }

    agent.last_seen = step_counter;
    agent.interaction_count++;
    agent.reputation = std::max(0.0f, std::min(1.0f,
        agent.reputation * 0.95f + (agent.relationship_valence > 0 ? 0.05f : -0.02f)));
}

SocialInteraction SocialInteractionModel::simulate_interaction(
    int target_id,
    const std::vector<float>& self_action,
    const std::vector<float>& self_emotion,
    float self_intention)
{
    SocialInteraction interaction;
    interaction.initiator_id = 0;
    interaction.target_id = target_id;
    interaction.action_signal = self_action;
    interaction.timestamp = step_counter;

    interaction.response_signal.assign(8, 0.0f);

    if (target_id < 0 || target_id >= MAX_SOCIAL_AGENTS
        || !social_agents[target_id].is_active) {
        interaction.outcome = 0.0f;
        interaction.emotional_impact = 0.0f;
        interaction.trust_update = 0.0f;
        return interaction;
    }

    auto& target = social_agents[target_id];

    for (int d = 0; d < 8 && d < (int)self_action.size(); d++) {
        float act = self_action[d];
        float agent_resp = d < (int)target.recent_actions.size()
            ? target.recent_actions[d] * target.relationship_valence : 0.0f;

        interaction.response_signal[d] = act * 0.3f
            + agent_resp * 0.4f
            + target.reputation * 0.2f
            + reciprocity * 0.1f;
        interaction.response_signal[d] = std::max(-1.0f, std::min(1.0f,
            interaction.response_signal[d]));
    }

    float alignment = 0.0f;
    for (int d = 0; d < 8 && d < (int)self_action.size(); d++) {
        alignment += self_action[d] * interaction.response_signal[d];
    }
    alignment = std::max(-1.0f, std::min(1.0f, alignment / 4.0f));

    interaction.outcome = alignment * 0.5f
        + target.relationship_valence * 0.3f
        + self_intention * 0.2f;

    float emo_self = 0.0f, emo_other = 0.0f;
    for (int d = 0; d < 4 && d < (int)self_emotion.size(); d++)
        emo_self += self_emotion[d];
    for (int d = 0; d < 4; d++)
        emo_other += target.emotional_state[d];
    interaction.emotional_impact = (emo_other / 4.0f - emo_self / 4.0f) * 0.5f;

    interaction.trust_update = alignment * 0.1f;

    return interaction;
}

void SocialInteractionModel::learn_from_interaction(
    const SocialInteraction& interaction,
    float reward)
{
    interaction_history.push_back(interaction);
    if ((int)interaction_history.size() > INTERACTION_HISTORY * 2)
        interaction_history.pop_front();

    int target_id = interaction.target_id;
    if (target_id < 0 || target_id >= MAX_SOCIAL_AGENTS) return;

    auto& agent = social_agents[target_id];

    agent.relationship_valence = agent.relationship_valence * 0.9f
        + interaction.outcome * 0.1f;

    agent.reciprocity_score = agent.reciprocity_score * 0.85f
        + (interaction.outcome > 0.3f ? 0.15f : 0.0f);

    agent.reputation = agent.reputation * 0.9f
        + (reward > 0.0f ? 0.1f : -0.05f);
    agent.reputation = std::max(0.0f, std::min(1.0f, agent.reputation));

    social_satisfaction = social_satisfaction * 0.9f
        + std::max(0.0f, interaction.outcome) * 0.1f;

    for (int d = 0; d < 32; d++) {
        float val = d < (int)interaction.action_signal.size()
            ? interaction.action_signal[d] : 0.0f;
        social_embedding[d] = social_embedding[d] * 0.95f + val * 0.05f;
    }

    _update_group_identity();
    _compute_social_metrics();
}

void SocialInteractionModel::update_relationships(float time_decay) {
    for (int i = 0; i < n_active_agents; i++) {
        if (!social_agents[i].is_active) continue;

        uint64_t elapsed = step_counter - social_agents[i].last_seen;
        if (elapsed > 1000) {
            float decay = time_decay * (float)elapsed / 1000.0f;
            social_agents[i].relationship_valence *= (1.0f - std::min(0.5f, decay));
            social_agents[i].reputation *= (1.0f - decay * 0.3f);
        }
    }
}

std::vector<float> SocialInteractionModel::get_social_context_vector() const {
    std::vector<float> ctx(32, 0.0f);

    for (int d = 0; d < 32; d++) {
        ctx[d] = social_embedding[d] * 0.4f
            + (d < IDENTITY_DIM ? group_identity[d] : 0.0f) * 0.3f
            + social_cohesion * 0.2f + social_satisfaction * 0.1f;
    }

    return ctx;
}

std::vector<float> SocialInteractionModel::get_affiliation_vector() const {
    std::vector<float> af(16, 0.0f);

    for (int i = 0; i < std::min(n_active_agents, 3); i++) {
        if (!social_agents[i].is_active) continue;
        float weight = social_agents[i].relationship_valence * 0.5f + 0.5f;
        for (int d = 0; d < 16 && d < IDENTITY_DIM; d++) {
            af[d] += social_agents[i].identity[d % IDENTITY_DIM] * weight;
        }
    }

    float norm = 0.0f;
    for (auto v : af) norm += v * v;
    if (norm > 1e-6f) {
        norm = std::sqrt(norm);
        for (auto& v : af) v = v / norm * 0.5f;
    }

    return af;
}

std::string SocialInteractionModel::get_relationship_summary() const {
    std::ostringstream oss;
    oss << "n" << n_active_agents;

    int pos_count = 0, neg_count = 0;
    for (int i = 0; i < n_active_agents; i++) {
        if (social_agents[i].is_active) {
            if (social_agents[i].relationship_valence > 0.3f) pos_count++;
            else if (social_agents[i].relationship_valence < -0.3f) neg_count++;
        }
    }

    if (pos_count > neg_count) oss << "_friendly";
    else if (neg_count > pos_count) oss << "_tense";
    else oss << "_neutral";

    oss << "_r" << (int)(reciprocity * 100);
    oss << "_c" << (int)(social_cohesion * 100);

    return oss.str();
}

float SocialInteractionModel::get_social_confidence() const {
    if (n_active_agents == 0) return 0.3f;

    float total_valence = 0.0f;
    int active_count = 0;
    for (int i = 0; i < n_active_agents; i++) {
        if (social_agents[i].is_active) {
            total_valence += social_agents[i].relationship_valence;
            active_count++;
        }
    }

    float avg_valence = active_count > 0 ? total_valence / (float)active_count : 0.0f;
    return 0.3f + (avg_valence + 1.0f) * 0.35f;
}

std::string SocialInteractionModel::get_group_description() const {
    std::ostringstream oss;
    if (social_cohesion > 0.6f) oss << "cohesive";
    else if (social_cohesion > 0.3f) oss << "loose";
    else oss << "fragmented";

    if (reciprocity > 0.5f) oss << "_reciprocal";
    else oss << "_self_serving";

    return oss.str();
}

std::vector<float> SocialInteractionModel::predict_agent_response(
    int agent_id,
    const std::vector<float>& action) const
{
    std::vector<float> response(8, 0.0f);
    if (agent_id < 0 || agent_id >= MAX_SOCIAL_AGENTS
        || !social_agents[agent_id].is_active)
        return response;

    const auto& agent = social_agents[agent_id];
    for (int d = 0; d < 8; d++) {
        float act = d < (int)action.size() ? action[d] : 0.0f;
        float hist = d < (int)agent.recent_actions.size() ? agent.recent_actions[d] : 0.0f;

        response[d] = act * (0.3f + agent.reciprocity_score * 0.2f)
            + hist * (0.2f + agent.relationship_valence * 0.15f);
        response[d] = std::max(-1.0f, std::min(1.0f, response[d]));
    }

    return response;
}

void SocialInteractionModel::_update_group_identity() {
    std::vector<float> acc(IDENTITY_DIM, 0.0f);
    int count = 0;
    for (int i = 0; i < n_active_agents; i++) {
        if (social_agents[i].is_active) {
            for (int d = 0; d < IDENTITY_DIM; d++) {
                acc[d] += social_agents[i].identity[d];
            }
            count++;
        }
    }
    if (count > 0) {
        for (int d = 0; d < IDENTITY_DIM; d++) {
            float avg = acc[d] / (float)count;
            group_identity[d] = group_identity[d] * 0.9f + avg * 0.1f;
        }
    }
}

void SocialInteractionModel::_compute_social_metrics() {
    if (n_active_agents < 1) return;

    float sim_sum = 0.0f;
    int pairs = 0;
    for (int i = 0; i < n_active_agents; i++) {
        if (!social_agents[i].is_active) continue;
        for (int j = i + 1; j < n_active_agents; j++) {
            if (!social_agents[j].is_active) continue;
            sim_sum += _cosine_sim(
                social_agents[i].identity,
                social_agents[j].identity);
            pairs++;
        }
    }

    float avg_sim = pairs > 0 ? std::max(0.0f, sim_sum / (float)pairs) : 0.0f;
    social_cohesion = social_cohesion * 0.9f + avg_sim * 0.1f;
}

float SocialInteractionModel::_cosine_sim(
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
    return dot / denom;
}