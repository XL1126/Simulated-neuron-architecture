#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

struct CreativeIdea {
    std::vector<float> idea_vector;
    std::vector<float> source_concepts;
    float novelty;
    float usefulness;
    float surprise;
    float coherence;
    float total_quality;
    uint64_t created_at;
    int generation_mode;
};

class CreativeGenerator {
public:
    static constexpr int IDEA_DIM = 32;
    static constexpr int CONCEPT_DIM = 32;
    static constexpr int MAX_IDEAS = 16;
    static constexpr int BLEND_HISTORY = 24;

    CreativeIdea idea_pool[MAX_IDEAS];
    int n_ideas;
    int best_idea_idx;
    std::vector<float> creative_context;
    std::vector<float> conceptual_blends;
    float creativity_level;
    float divergent_thinking;
    float convergent_thinking;
    float fluency;
    float originality;
    uint64_t step_counter;

    CreativeGenerator();

    void init(uint32_t seed);

    void generate_ideas(
        const std::vector<float>& concept_activities,
        const std::vector<float>& self_state,
        const std::vector<float>& emotion_vector,
        const std::vector<float>& episodic_context,
        const std::vector<float>& temporal_context,
        float spontaneity,
        float dopamine);

    void evaluate_ideas(
        const std::vector<float>& self_state,
        const std::vector<float>& world_state,
        float meta_confidence);

    void select_best_idea(float randomness);

    std::vector<float> get_best_idea_vector() const;
    std::string get_idea_description() const;

    std::vector<float> get_creative_modulation(
        const std::vector<float>& concept_activities,
        int n_concepts) const;

    std::vector<float> get_novel_concept_combination(int n_concepts) const;

    std::string get_creativity_summary() const;

private:
    void _blend_concepts(
        const std::vector<float>& c1,
        const std::vector<float>& c2,
        std::vector<float>& result,
        float blend_ratio);

    float _compute_novelty(
        const std::vector<float>& idea,
        const std::deque<std::vector<float>>& history) const;

    std::deque<std::vector<float>> idea_history;
    float _cosine_sim(const std::vector<float>& a, const std::vector<float>& b) const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};