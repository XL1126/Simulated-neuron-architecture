#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

struct SocialAgent {
    std::vector<float> identity;
    std::vector<float> recent_actions;
    std::vector<float> emotional_state;
    float reputation;
    float relationship_valence;
    float reciprocity_score;
    float similarity;
    int interaction_count;
    uint64_t last_seen;
    int agent_id;
    bool is_active;
};

struct SocialInteraction {
    int initiator_id;
    int target_id;
    std::vector<float> action_signal;
    std::vector<float> response_signal;
    float outcome;
    float emotional_impact;
    float trust_update;
    uint64_t timestamp;
};

class SocialInteractionModel {
public:
    static constexpr int MAX_SOCIAL_AGENTS = 6;
    static constexpr int IDENTITY_DIM = 16;
    static constexpr int INTERACTION_HISTORY = 32;

    SocialAgent social_agents[MAX_SOCIAL_AGENTS];
    int n_active_agents;
    std::deque<SocialInteraction> interaction_history;
    std::vector<float> social_embedding;
    std::vector<float> group_identity;
    float social_cohesion;
    float social_influence;
    float reciprocity;
    float social_learning_rate;
    float affiliation_need;
    float social_satisfaction;
    uint64_t step_counter;

    SocialInteractionModel();

    void init(uint32_t seed);

    void add_or_update_agent(
        int agent_id,
        const std::vector<float>& observed_identity,
        const std::vector<float>& observed_action,
        const std::vector<float>& observed_emotion);

    SocialInteraction simulate_interaction(
        int target_id,
        const std::vector<float>& self_action,
        const std::vector<float>& self_emotion,
        float self_intention);

    void learn_from_interaction(
        const SocialInteraction& interaction,
        float reward);

    void update_relationships(float time_decay);

    std::vector<float> get_social_context_vector() const;

    std::vector<float> get_affiliation_vector() const;

    std::string get_relationship_summary() const;

    float get_social_confidence() const;

    std::string get_group_description() const;

    std::vector<float> predict_agent_response(
        int agent_id,
        const std::vector<float>& action) const;

private:
    void _update_group_identity();
    void _compute_social_metrics();
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};