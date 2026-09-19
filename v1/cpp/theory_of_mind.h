#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>
#include <deque>
#include <algorithm>

struct OtherAgent {
    std::vector<float> belief_state;
    std::vector<float> desire_state;
    std::vector<float> intention_state;
    std::vector<float> perceived_location;
    std::vector<float> recent_actions;
    float familiarity;
    float trust_level;
    float perceived_competence;
    uint64_t last_interaction;
    int identity_id;
    bool is_present;
};

struct SocialInference {
    int agent_id;
    std::vector<float> predicted_action;
    float prediction_confidence;
    float empathy_signal;
    bool false_belief_detected;
    std::vector<float> perspective_taking;
};

struct SocialContext {
    std::vector<float> group_norm;
    std::vector<float> social_hierarchy;
    float cooperation_signal;
    float competition_signal;
    float social_anxiety;
};

class TheoryOfMind {
public:
    static constexpr int MAX_AGENTS = 4;
    static constexpr int BELIEF_DIM = 32;
    static constexpr int DESIRE_DIM = 16;
    static constexpr int INTENTION_DIM = 16;
    static constexpr int SOCIAL_CONTEXT_DIM = 16;

    OtherAgent agents[MAX_AGENTS];
    int n_agents;
    SocialContext social_ctx;
    std::vector<float> self_other_boundary;
    float self_other_separation;
    float false_belief_accuracy;
    float empathy_level;
    float social_awareness;
    uint64_t step_counter;

    TheoryOfMind();

    void init(uint32_t seed);

    void perceive_agent(
        int agent_id,
        const std::vector<float>& observed_state,
        const std::vector<float>& observed_action,
        float interaction_outcome);

    void update_beliefs(
        int agent_id,
        const std::vector<float>& world_state,
        const std::vector<float>& self_state);

    void infer_intentions(
        int agent_id,
        const std::vector<float>& recent_history);

    SocialInference predict_other(
        int agent_id,
        const std::vector<float>& current_world,
        const std::vector<float>& self_intention);

    bool detect_false_belief(
        int agent_id,
        const std::vector<float>& ground_truth);

    std::vector<float> take_perspective(
        int agent_id,
        const std::vector<float>& situation) const;

    void update_self_other_boundary(
        const std::vector<float>& self_state,
        const std::vector<float>& other_observation);

    void update_social_context(
        const std::vector<float>& interaction_valence,
        float cooperation_level,
        float reward_shared);

    std::vector<float> get_social_modulation() const;
    std::vector<float> get_empathy_vector(int agent_id) const;
    std::string get_social_narrative() const;
    float get_theory_of_mind_level() const;

private:
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    void _normalize(std::vector<float>& v) const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};