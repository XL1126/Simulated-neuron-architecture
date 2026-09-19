#include "language_sequence.h"
#include <random>

static inline bool is_bad(float x) { return !std::isfinite(x); }

LanguageSequenceGenerator::LanguageSequenceGenerator() {}

void LanguageSequenceGenerator::init() {
    seq_context.assign(CONTEXT_DIM, 0.0f);
    energy_level = 0.1f;
    pattern_freshness = 0.0f;
    output_variety_count = 0;
    last_best_idx = -1;
}

std::vector<std::pair<int, float>> LanguageSequenceGenerator::generate_sequence(
    const std::vector<float>& thought_vec,
    const std::vector<float>& narrative_vec,
    float meta_valence, float meta_arousal, float meta_conf,
    const std::vector<float>& embeddings,
    int vocab_size, uint64_t step) const {

    std::vector<std::pair<int, float>> result;
    if (vocab_size <= 0) return result;

    std::vector<float> query(CONTEXT_DIM, 0.0f);
    for (int i = 0; i < 6 && i < (int)thought_vec.size(); i++)
        query[i] = thought_vec[i] * 0.3f;
    for (int i = 0; i < 4 && i < (int)narrative_vec.size(); i++)
        query[6 + i] = narrative_vec[i] * 0.2f;
    query[10] = meta_valence * 0.5f;
    query[11] = meta_arousal * 0.5f;
    query[12] = (1.0f - meta_conf) * 0.5f;
    for (int i = 0; i < CONTEXT_DIM; i++) {
        seq_context[i] = seq_context[i] * 0.5f + query[i] * 0.5f;
    }

    int words_to_gen = 1;
    if (meta_arousal > 0.25f) words_to_gen = 2;
    if (meta_arousal > 0.4f || (meta_conf < 0.5f && meta_arousal > 0.2f)) words_to_gen = 3;
    if (meta_arousal > 0.55f && meta_conf < 0.4f) words_to_gen = 4;
    words_to_gen = std::min(words_to_gen, MAX_WORDS);

    int embed_dim = vocab_size > 0 ? (int)embeddings.size() / vocab_size : 32;
    if (embed_dim < 1) embed_dim = 32;

    std::vector<float> current_query = seq_context;

    float temperature = 0.3f + meta_arousal * 3.0f + (1.0f - meta_conf) * 2.0f;

    for (int w = 0; w < words_to_gen; w++) {
        struct Candidate { int idx; float score; };
        std::vector<Candidate> candidates;
        float max_score = -1e10f;

        for (int v = 0; v < vocab_size; v++) {
            float score = 0.0f;
            for (int d = 0; d < std::min((int)current_query.size(), embed_dim); d++) {
                float emb = embeddings[v * embed_dim + d];
                float q = current_query[d];
                if (!is_bad(emb) && !is_bad(q)) score += emb * q;
            }
            for (auto& prev : result) {
                if (prev.first == v) score -= 0.3f;
            }
            float step_noise = (float)((v * 1103515245 + (int)step * 25214903917ULL) % 10000) / 10000.0f;
            score += (step_noise - 0.5f) * temperature * 0.6f;
            candidates.push_back({v, score});
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (auto& c : candidates) {
            c.score = std::exp((c.score - max_score) / (temperature + 0.1f));
            sum_exp += c.score;
        }

        float roll = (float)((step * 7 + w * 3 + 1) * 1103515245 % 10000) / 10000.0f;
        float cum = 0.0f;
        int pick = candidates[0].idx;
        for (auto& c : candidates) {
            cum += c.score / sum_exp;
            if (roll <= cum) { pick = c.idx; break; }
        }

        result.push_back({pick, candidates[pick].score});

        for (int d = 0; d < std::min((int)current_query.size(), embed_dim); d++) {
            float emb = embeddings[pick * embed_dim + d];
            current_query[d] = current_query[d] * 0.7f + emb * 0.3f;
        }
    }

    return result;
}