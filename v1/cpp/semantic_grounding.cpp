#include "semantic_grounding.h"
#include <random>

SemanticGrounding::SemanticGrounding()
    : association_strength(0.0f), validation_accuracy(0.3f),
      n_concepts(0), embed_dim(0), n_associations_formed(0),
      base_lr(0.01f)
{}

void SemanticGrounding::init(int n_concepts, int embed_dim, uint32_t seed) {
    this->n_concepts = n_concepts;
    this->embed_dim = embed_dim;

    concept_sensory_asso.assign(n_concepts * GROUND_DIM, 0.0f);
    sensory_concept_asso.assign(GROUND_DIM * n_concepts, 0.0f);
    grounded_embeddings.assign(n_concepts * EMBED_DIM, 0.0f);
    concept_value_bias.assign(n_concepts, 0.0f);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (int c = 0; c < n_concepts; c++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            grounded_embeddings[c * EMBED_DIM + d] = dist(rng);
        }
        float norm = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
            norm += grounded_embeddings[c * EMBED_DIM + d] * grounded_embeddings[c * EMBED_DIM + d];
        if (norm > 1e-10f) {
            norm = std::sqrt(norm);
            for (int d = 0; d < EMBED_DIM; d++)
                grounded_embeddings[c * EMBED_DIM + d] /= norm;
        }
    }

    association_strength = 0.0f;
    validation_accuracy = 0.3f;
    n_associations_formed = 0;
}

void SemanticGrounding::assoc_learn(
    const std::vector<float>& sensory_features,
    const std::vector<float>& concept_activities,
    float reward_signal,
    float self_model_error)
{
    if (sensory_features.empty() || concept_activities.empty()) return;

    std::vector<float> compressed(GROUND_DIM, 0.0f);
    size_t seg = std::max((size_t)1, sensory_features.size() / GROUND_DIM);
    for (int d = 0; d < GROUND_DIM; d++) {
        float acc = 0.0f;
        int cnt = 0;
        for (size_t i = d * seg; i < (d + 1) * seg && i < sensory_features.size(); i++) {
            if (!is_bad(sensory_features[i])) {
                acc += sensory_features[i];
                cnt++;
            }
        }
        compressed[d] = cnt > 0 ? acc / (float)cnt : 0.0f;
    }

    float lr = base_lr * (0.5f + std::abs(reward_signal) * 1.5f
        + self_model_error * 0.5f);

    for (int c = 0; c < n_concepts && c < (int)concept_activities.size(); c++) {
        float ca = concept_activities[c];
        if (ca < 0.1f) continue;

        for (int d = 0; d < GROUND_DIM; d++) {
            float& wt = concept_sensory_asso[c * GROUND_DIM + d];
            wt = wt * (1.0f - lr * 0.5f) + compressed[d] * ca * lr;
            wt = std::max(-0.5f, std::min(0.5f, wt));
        }
    }

    for (int d = 0; d < GROUND_DIM; d++) {
        float sd = compressed[d];
        if (std::abs(sd) < 0.01f) continue;
        for (int c = 0; c < n_concepts; c++) {
            float& wt = sensory_concept_asso[d * n_concepts + c];
            wt = wt * (1.0f - lr * 0.3f) + sd * concept_activities[c] * lr * 0.5f;
            wt = std::max(-0.5f, std::min(0.5f, wt));
        }
    }

    for (int c = 0; c < n_concepts && c < (int)concept_activities.size(); c++) {
        concept_value_bias[c] = concept_value_bias[c] * 0.995f
            + reward_signal * concept_activities[c] * 0.005f;
        concept_value_bias[c] = std::max(-1.0f, std::min(1.0f, concept_value_bias[c]));
    }

    float total_wt = 0.0f;
    for (auto v : concept_sensory_asso) total_wt += std::abs(v);
    total_wt /= (float)(n_concepts * GROUND_DIM + 1);
    association_strength = std::min(1.0f, total_wt * 5.0f);

    n_associations_formed = (int)(association_strength * 1000.0f);

    validation_accuracy = 0.3f + association_strength * 0.5f;
}

std::vector<float> SemanticGrounding::predict_concepts(
    const std::vector<float>& sensory_features) const
{
    if (n_concepts <= 0) return {};

    std::vector<float> compressed(GROUND_DIM, 0.0f);
    if (!sensory_features.empty()) {
        size_t seg = std::max((size_t)1, sensory_features.size() / GROUND_DIM);
        for (int d = 0; d < GROUND_DIM; d++) {
            float acc = 0.0f;
            int cnt = 0;
            for (size_t i = d * seg; i < (d + 1) * seg && i < sensory_features.size(); i++) {
                if (!is_bad(sensory_features[i])) {
                    acc += sensory_features[i];
                    cnt++;
                }
            }
            compressed[d] = cnt > 0 ? acc / (float)cnt : 0.0f;
        }
    }

    std::vector<float> result(n_concepts, 0.0f);
    for (int c = 0; c < n_concepts; c++) {
        float score = 0.0f;
        for (int d = 0; d < GROUND_DIM; d++) {
            score += compressed[d] * sensory_concept_asso[d * n_concepts + c]
                + compressed[d] * concept_sensory_asso[c * GROUND_DIM + d];
        }
        score += concept_value_bias[c] * 0.3f;
        result[c] = std::tanh(score * 1.5f) * 0.5f + 0.5f;
    }
    return result;
}

std::vector<float> SemanticGrounding::get_grounded_concepts(
    const std::vector<float>& sensory_features,
    const std::vector<float>& concept_activities,
    float mix_ratio) const
{
    if (n_concepts <= 0) return {};

    auto predicted = predict_concepts(sensory_features);

    std::vector<float> result(n_concepts, 0.0f);
    for (int c = 0; c < n_concepts; c++) {
        float pred_val = (c < (int)predicted.size()) ? predicted[c] : 0.0f;
        float orig_val = (c < (int)concept_activities.size()) ? concept_activities[c] : 0.0f;
        float bias = (c < (int)concept_value_bias.size()) ? concept_value_bias[c] : 0.0f;

        float alpha = mix_ratio * (0.3f + association_strength * 0.7f);
        result[c] = orig_val * (1.0f - alpha) + pred_val * alpha + bias * 0.1f;
        result[c] = std::max(0.0f, std::min(1.0f, result[c]));
    }
    return result;
}

void SemanticGrounding::decay(float rate) {
    for (auto& v : concept_sensory_asso) v *= (1.0f - rate);
    for (auto& v : sensory_concept_asso) v *= (1.0f - rate);
    for (auto& v : concept_value_bias) v *= (1.0f - rate * 0.5f);
}