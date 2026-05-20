#include "language_layer.h"

LanguageLayer::LanguageLayer()
    : vocab_size(0), embed_dim(0), lang_neurons(0),
      adaptive_threshold(BASE_THRESHOLD), concept_mean(0.01f), concept_std(0.01f)
{
    rng.seed(42);
    state.syntax_phase = 0.0f;
    state.stack_depth = 0;
    state.last_output = "";
    state.repeat_count = 0;
}

LanguageLayer::LanguageLayer(size_t vocab_size, size_t embed_dim, size_t lang_neurons)
    : vocab_size(vocab_size), embed_dim(embed_dim), lang_neurons(lang_neurons),
      adaptive_threshold(BASE_THRESHOLD), concept_mean(0.01f), concept_std(0.01f)
{
    rng.seed(42);
    embedding_matrix.resize(vocab_size * embed_dim, 0.0f);
    projection_weights.resize(embed_dim * lang_neurons, 0.0f);
    syntax_action_weights.resize(ACT_COUNT * embed_dim, 0.0f);

    state.syntax_phase = 0.0f;
    state.stack_depth = 0;
    state.last_output = "";
    state.repeat_count = 0;
    state.stack_embedding.resize(embed_dim, 0.0f);
    state.context_embedding.resize(embed_dim, 0.0f);
    for (int i = 0; i < ACT_COUNT; i++) state.action_probs[i] = 0.0f;
}

void LanguageLayer::init(size_t vocab_size, size_t embed_dim, size_t lang_neurons) {
    this->vocab_size = vocab_size;
    this->embed_dim = embed_dim;
    this->lang_neurons = lang_neurons;
    embedding_matrix.resize(vocab_size * embed_dim, 0.0f);
    projection_weights.resize(embed_dim * lang_neurons, 0.0f);
    syntax_action_weights.resize(ACT_COUNT * embed_dim, 0.0f);
    state.stack_embedding.resize(embed_dim, 0.0f);
    state.context_embedding.resize(embed_dim, 0.0f);
}

void LanguageLayer::initialize(const std::vector<std::string>& vocabulary) {
    vocab_words = vocabulary;
    if (vocab_words.size() > vocab_size)
        vocab_words.resize(vocab_size);

    for (size_t i = 0; i < vocab_words.size(); i++) {
        word_to_idx[vocab_words[i]] = i;
    }

    std::normal_distribution<float> embed_dist(0.0f, 1.0f / std::sqrt((float)embed_dim));
    for (size_t i = 0; i < vocab_words.size(); i++) {
        for (size_t d = 0; d < embed_dim; d++) {
            embedding_matrix[i * embed_dim + d] = embed_dist(rng);
        }
        float norm = 0.0f;
        for (size_t d = 0; d < embed_dim; d++)
            norm += embedding_matrix[i * embed_dim + d] * embedding_matrix[i * embed_dim + d];
        if (norm > 1e-10f) {
            norm = std::sqrt(norm);
            for (size_t d = 0; d < embed_dim; d++)
                embedding_matrix[i * embed_dim + d] /= norm;
        }
    }

    std::normal_distribution<float> proj_dist(0.0f, 0.1f / std::sqrt((float)embed_dim));
    for (size_t e = 0; e < embed_dim; e++) {
        for (size_t n = 0; n < lang_neurons; n++) {
            projection_weights[e * lang_neurons + n] = proj_dist(rng);
        }
    }

    for (int a = 0; a < ACT_COUNT; a++) {
        for (size_t d = 0; d < embed_dim; d++) {
            syntax_action_weights[a * embed_dim + d] = 0.0f;
        }
    }
}

std::vector<float> LanguageLayer::project_neural_activity(
    const std::vector<float>& lang_activity) {
    std::vector<float> embedding(embed_dim, 0.0f);
    if (lang_activity.empty()) return embedding;

    size_t n_per_dim = std::max((size_t)1, lang_neurons / embed_dim);
    for (size_t d = 0; d < embed_dim; d++) {
        for (size_t n = 0; n < std::min(n_per_dim, lang_activity.size()); n++) {
            size_t nidx = std::min(d * n_per_dim + n, lang_activity.size() - 1);
            size_t widx = std::min(d * lang_neurons + nidx, projection_weights.size() - 1);
            embedding[d] += lang_activity[nidx] * projection_weights[widx];
        }
    }

    float norm = 0.0f;
    for (auto v : embedding) norm += v * v;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& v : embedding) v /= norm;
    }
    return embedding;
}

std::vector<std::pair<std::string, float>> LanguageLayer::retrieve_top_k(
    const std::vector<float>& query_embedding, size_t k) {
    if (query_embedding.empty() || vocab_words.empty()) return {};

    std::vector<std::pair<size_t, float>> scores;
    for (size_t i = 0; i < vocab_words.size(); i++) {
        std::vector<float> word_emb(embed_dim);
        for (size_t d = 0; d < embed_dim; d++)
            word_emb[d] = embedding_matrix[i * embed_dim + d];
        float sim = _cosine_similarity(query_embedding, word_emb);
        scores.emplace_back(i, sim);
    }

    std::sort(scores.begin(), scores.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    std::vector<std::pair<std::string, float>> result;
    for (size_t i = 0; i < std::min(k, scores.size()); i++) {
        result.emplace_back(vocab_words[scores[i].first], scores[i].second);
    }
    return result;
}

std::string LanguageLayer::generate_output(
    float dopamine, float norepinephrine,
    float self_model_error, float curiosity,
    const std::vector<float>& concept_activity,
    const std::vector<float>& lang_activity,
    uint64_t step) {

    std::vector<float> query_embedding = project_neural_activity(lang_activity);

    float novelty_bonus = norepinephrine * 0.2f + curiosity * 0.15f;
    std::uniform_real_distribution<float> noise_dist(-0.5f, 0.5f);

    for (size_t d = 0; d < embed_dim; d++) {
        query_embedding[d] += state.context_embedding[d] * 0.35f
                             + state.stack_embedding[d] * 0.1f;
        float step_noise = (float)((d * 1103515245 + step * 25214903917ULL) % 10000) / 10000.0f;
        query_embedding[d] += (step_noise - 0.5f) * (norepinephrine * 0.15f + curiosity * 0.10f + 0.03f);
    }

    float norm = 0.0f;
    for (auto v : query_embedding) norm += v * v;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& v : query_embedding) v /= norm;
    }

    float mean_act = 0.0f;
    for (auto a : concept_activity) mean_act += a;
    mean_act /= std::max(1.0f, (float)concept_activity.size());

    float var_act = 0.0f;
    for (auto a : concept_activity) var_act += (a - mean_act) * (a - mean_act);
    var_act /= std::max(1.0f, (float)concept_activity.size());

    concept_mean = concept_mean * (1.0f - ADAPTIVE_THRESHOLD_ALPHA)
                   + mean_act * ADAPTIVE_THRESHOLD_ALPHA;
    concept_std = concept_std * (1.0f - ADAPTIVE_THRESHOLD_ALPHA)
                  + std::sqrt(std::max(0.0001f, var_act)) * ADAPTIVE_THRESHOLD_ALPHA;

    adaptive_threshold = std::max(BASE_THRESHOLD,
        concept_mean + 1.5f * concept_std);

    float spk_thresh = adaptive_threshold * (1.0f - dopamine * 0.5f
                        - self_model_error * 0.2f);

    auto candidates = retrieve_top_k(query_embedding, 15);
    if (candidates.empty()) {
        if (!vocab_words.empty()) return vocab_words[0];
        return "...";
    }

    _update_syntax_action_probs();

    int action = 0;
    float max_prob = state.action_probs[0];
    for (int a = 1; a < ACT_COUNT; a++) {
        if (state.action_probs[a] > max_prob) {
            max_prob = state.action_probs[a];
            action = a;
        }
    }

    std::string selected_word;

    if (action == ACT_GEN && candidates[0].second > spk_thresh) {
        float temperature = 0.15f + norepinephrine * 5.0f + curiosity * 3.0f;

        struct Weighted { size_t i; float w; };
        std::vector<Weighted> candidate_pool;

        for (size_t i = 0; i < std::min((size_t)8, candidates.size()); i++) {
            float w = candidates[i].second;
            bool is_repeat = (candidates[i].first == state.last_output);
            if (is_repeat) {
                if (state.repeat_count > 0) {
                    w *= 0.05f;
                } else {
                    w *= 0.25f;
                }
            }
            float noise_jitter = (float)((rng() % 100) - 50) / 200.0f;
            float step_jitter = (float)((i * 1103515245 + step * 25214903917ULL) % 10000) / 10000.0f;
            w += (noise_jitter + step_jitter - 0.5f) * temperature * 0.5f;
            candidate_pool.push_back({i, w});
        }

        float max_w = candidate_pool[0].w;
        for (auto& cp : candidate_pool) {
            if (cp.w > max_w) max_w = cp.w;
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(candidate_pool.size());
        for (size_t i = 0; i < candidate_pool.size(); i++) {
            probs[i] = std::exp((candidate_pool[i].w - max_w) / std::max(0.1f, temperature));
            sum_exp += probs[i];
        }

        if (sum_exp > 1e-10f) {
            for (auto& p : probs) p /= sum_exp;
            std::uniform_real_distribution<float> roll_dist(0.0f, 1.0f);
            float roll = roll_dist(rng);
            float cum = 0.0f;
            size_t pick = 0;
            for (size_t i = 0; i < probs.size(); i++) {
                cum += probs[i];
                if (roll <= cum) { pick = i; break; }
            }
            selected_word = candidates[candidate_pool[pick].i].first;
        } else {
            selected_word = candidates[0].first;
        }
    } else if (candidates[0].second > spk_thresh) {
        selected_word = candidates[0].first;
    } else {
        float best_act = 0.0f;
        size_t best_i = 0;
        for (size_t i = 0; i < concept_activity.size() && i < vocab_words.size(); i++) {
            if (concept_activity[i] > best_act) {
                best_act = concept_activity[i];
                best_i = i;
            }
        }
        if (best_act > BASE_THRESHOLD * 0.5f && best_i < vocab_words.size()) {
            selected_word = vocab_words[best_i];
        } else {
            selected_word = candidates[0].first;
        }
    }

    if (selected_word == state.last_output) {
        state.repeat_count++;
    } else {
        state.repeat_count = 0;
    }
    state.last_output = selected_word;

    if (!vocab_words.empty() && selected_word != vocab_words[0]) {
        size_t w_idx = 0;
        auto it = word_to_idx.find(selected_word);
        if (it != word_to_idx.end()) w_idx = it->second;

        for (size_t d = 0; d < embed_dim; d++) {
            state.context_embedding[d] = state.context_embedding[d] * 0.8f
                + embedding_matrix[w_idx * embed_dim + d] * 0.2f;
        }
    }

    update_syntax_state(dopamine, norepinephrine);

    return selected_word;
}

void LanguageLayer::inject_dmn_context(const std::vector<float>& dmn_ctx, float spontaneity) {
    float alpha = spontaneity * 0.3f;
    for (size_t d = 0; d < std::min(embed_dim, dmn_ctx.size()); d++) {
        state.context_embedding[d] = state.context_embedding[d] * (1.0f - alpha)
            + dmn_ctx[d] * alpha;
    }
}

void LanguageLayer::update_projection_weights(
    const std::vector<float>& lang_activity,
    const std::vector<float>& target_embedding, float lr) {
    if (lang_activity.size() < embed_dim || target_embedding.size() < embed_dim)
        return;

    size_t n_per_dim = std::max((size_t)1, lang_neurons / embed_dim);
    for (size_t d = 0; d < embed_dim; d++) {
        for (size_t n = 0; n < n_per_dim; n++) {
            size_t nidx = std::min(d * n_per_dim + n, lang_activity.size() - 1);
            size_t widx = d * lang_neurons + nidx;
            if (widx < projection_weights.size()) {
                float pred = lang_activity[nidx] * projection_weights[widx];
                float err = target_embedding[d] - pred;
                projection_weights[widx] += lr * err * lang_activity[nidx];
                projection_weights[widx] = std::max(-0.5f,
                    std::min(0.5f, projection_weights[widx]));
            }
        }
    }
}

void LanguageLayer::apply_self_feedback(const std::string& output_word,
                                         std::vector<float>& lang_activity) {
    auto it = word_to_idx.find(output_word);
    if (it == word_to_idx.end() || lang_activity.empty()) return;

    size_t w_idx = it->second;
    size_t n_per_dim = std::max((size_t)1, lang_neurons / embed_dim);

    for (size_t d = 0; d < embed_dim; d++) {
        float emb_val = embedding_matrix[w_idx * embed_dim + d];
        for (size_t n = 0; n < n_per_dim; n++) {
            size_t nidx = std::min(d * n_per_dim + n, lang_activity.size() - 1);
            lang_activity[nidx] = std::max(0.0f, std::min(1.0f,
                lang_activity[nidx] + emb_val * 0.08f));
        }
    }

    for (size_t d = 0; d < std::min(embed_dim, lang_activity.size()); d++) {
        lang_activity[d] = std::max(0.0f, std::min(1.0f,
            lang_activity[d] + state.context_embedding[d] * 0.05f));
    }
}

void LanguageLayer::update_syntax_state(float dopamine, float norepinephrine) {
    state.syntax_phase += 0.05f + dopamine * 0.03f + norepinephrine * 0.02f;
    if (state.syntax_phase > 6.283f) state.syntax_phase -= 6.283f;

    if (norepinephrine > 0.4f) {
        state.stack_depth = std::max(0, state.stack_depth - 1);
    } else {
        state.stack_depth = std::min(3, state.stack_depth + 1);
    }

    for (size_t d = 0; d < state.stack_embedding.size(); d++) {
        state.stack_embedding[d] = state.stack_embedding[d] * 0.95f
            + std::sin(state.syntax_phase + (float)d * 0.3f) * 0.05f * (float)state.stack_depth;
    }
}

void LanguageLayer::_update_syntax_action_probs() {
    float raw[ACT_COUNT] = {0};

    for (int a = 0; a < ACT_COUNT; a++) {
        for (size_t d = 0; d < embed_dim; d++) {
            float sd = state.stack_embedding[d];
            float cd = state.context_embedding[d];
            float syn = state.syntax_phase;
            raw[a] += sd * syntax_action_weights[a * embed_dim + d] * 0.3f
                     + cd * syntax_action_weights[a * embed_dim + d] * 0.2f
                     + std::sin(syn + (float)a) * 0.1f;
        }
    }

    if (state.stack_depth == 0) {
        raw[ACT_NT_NP] += 0.5f;
        raw[ACT_REDUCE] -= 0.5f;
    } else if (state.stack_depth == 1) {
        raw[ACT_NT_VP] += 0.3f;
    } else if (state.stack_depth >= 2) {
        raw[ACT_REDUCE] += 0.4f;
        raw[ACT_NT_NP] -= 0.2f;
    }

    if (state.repeat_count > 1 && state.repeat_count <= 3) {
        raw[ACT_WAIT] += (float)state.repeat_count * 0.15f;
    }

    float max_raw = raw[0];
    for (int a = 1; a < ACT_COUNT; a++)
        if (raw[a] > max_raw) max_raw = raw[a];

    float sum_exp = 0.0f;
    for (int a = 0; a < ACT_COUNT; a++) {
        state.action_probs[a] = std::exp(raw[a] - max_raw);
        sum_exp += state.action_probs[a];
    }

    if (sum_exp > 1e-10f) {
        for (int a = 0; a < ACT_COUNT; a++)
            state.action_probs[a] /= sum_exp;
    }
}

float LanguageLayer::_cosine_similarity(const std::vector<float>& a,
                                         const std::vector<float>& b) const {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(std::max(1e-10f, norm_a * norm_b));
    return dot / denom;
}

std::string LanguageLayer::get_embedding_for_word(const std::string& word) const {
    auto it = word_to_idx.find(word);
    if (it == word_to_idx.end()) return "";

    size_t w_idx = it->second;
    std::string result;
    for (size_t d = 0; d < embed_dim; d++) {
        if (d > 0) result += ",";
        float val = embedding_matrix[w_idx * embed_dim + d];
        result += std::to_string(val);
    }
    return result;
}

float LanguageLayer::_hash_to_float(uint32_t h) {
    return (float)(h % 1000000) / 500000.0f - 1.0f;
}

void LanguageLayer::learn_syntax_weights(int best_action, float reward) {
    if (best_action < 0 || best_action >= ACT_COUNT) return;
    float lr = 0.01f * std::abs(reward);
    for (size_t d = 0; d < embed_dim; d++) {
        float sd = state.stack_embedding[d];
        float cd = state.context_embedding[d];
        float input = sd * 0.3f + cd * 0.2f;
        syntax_action_weights[best_action * embed_dim + d] += lr * input;
        syntax_action_weights[best_action * embed_dim + d] =
            std::max(-0.2f, std::min(0.2f, syntax_action_weights[best_action * embed_dim + d]));
    }
}