#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <deque>
#include <algorithm>

class SemanticGrounding {
public:
    static constexpr int GROUND_DIM = 32;
    static constexpr int MAX_CONCEPTS = 128;
    static constexpr int EMBED_DIM = 32;

    std::vector<float> concept_sensory_asso;
    std::vector<float> sensory_concept_asso;
    std::vector<float> grounded_embeddings;
    std::vector<float> concept_value_bias;
    float association_strength;
    float validation_accuracy;

    SemanticGrounding();

    void init(int n_concepts, int embed_dim, uint32_t seed);

    void assoc_learn(
        const std::vector<float>& sensory_features,
        const std::vector<float>& concept_activities,
        float reward_signal,
        float self_model_error);

    std::vector<float> predict_concepts(
        const std::vector<float>& sensory_features) const;

    std::vector<float> get_grounded_concepts(
        const std::vector<float>& sensory_features,
        const std::vector<float>& concept_activities,
        float mix_ratio) const;

    void decay(float rate);

    float get_association_strength() const { return association_strength; }

private:
    int n_concepts;
    int embed_dim;
    int n_associations_formed;
    float base_lr;

    static inline bool is_bad(float x) { return !std::isfinite(x); }
};