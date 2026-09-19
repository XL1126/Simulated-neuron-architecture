#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <random>
#include <deque>
#include <unordered_map>
#include <algorithm>

enum SyntaxAction {
    ACT_NT_NP = 0,
    ACT_NT_VP = 1,
    ACT_NT_PP = 2,
    ACT_GEN = 3,
    ACT_REDUCE = 4,
    ACT_WAIT = 5,
    ACT_COUNT = 6
};

struct LanguageState {
    float syntax_phase;
    std::vector<float> stack_embedding;
    int stack_depth;
    std::string last_output;
    int repeat_count;
    float action_probs[ACT_COUNT];
    std::vector<float> context_embedding;
};

class LanguageLayer {
public:
    LanguageLayer();
    LanguageLayer(size_t vocab_size, size_t embed_dim, size_t lang_neurons);

    void init(size_t vocab_size, size_t embed_dim, size_t lang_neurons);
    void initialize(const std::vector<std::string>& vocabulary);
    std::vector<float> project_neural_activity(const std::vector<float>& lang_activity);
    std::vector<std::pair<std::string, float>> retrieve_top_k(
        const std::vector<float>& query_embedding, size_t k);

    std::string generate_output(float dopamine, float norepinephrine,
                                 float self_model_error, float curiosity,
                                 const std::vector<float>& concept_activity,
                                 const std::vector<float>& lang_activity,
                                 uint64_t step);

    void inject_dmn_context(const std::vector<float>& dmn_ctx, float spontaneity);

    void update_projection_weights(const std::vector<float>& lang_activity,
                                    const std::vector<float>& target_embedding,
                                    float lr);
    void apply_self_feedback(const std::string& output_word,
                              std::vector<float>& lang_activity);
    void update_syntax_state(float dopamine, float norepinephrine);

    const LanguageState& get_state() const { return state; }
    const std::vector<float>& get_embedding_matrix() const { return embedding_matrix; }
    size_t get_vocab_size() const { return vocab_size; }
    size_t get_embed_dim() const { return embed_dim; }

    std::string get_embedding_for_word(const std::string& word) const;

    void learn_syntax_weights(int best_action, float reward);

private:
    size_t vocab_size;
    size_t embed_dim;
    size_t lang_neurons;
    std::vector<float> embedding_matrix;
    std::vector<float> projection_weights;
    std::vector<float> syntax_action_weights;
    std::vector<std::string> vocab_words;
    std::unordered_map<std::string, size_t> word_to_idx;
    LanguageState state;

    std::mt19937 rng;

    static constexpr float ADAPTIVE_THRESHOLD_ALPHA = 0.01f;
    static constexpr float BASE_THRESHOLD = 0.002f;
    float adaptive_threshold;
    float concept_mean;
    float concept_std;

    float _cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    void _update_syntax_action_probs();
    void _select_action();
    std::string _generate_word(const std::vector<float>& embed_query);
    static float _hash_to_float(uint32_t h);
};