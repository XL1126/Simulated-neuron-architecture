#include "state_word_mapper.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

StateToWordMapper::StateToWordMapper()
    : vocab_size(0), selected_word_idx(0), selected_score(0.0f),
      prev_valence(0.0f), prev_arousal(0.3f)
{}

void StateToWordMapper::init(int vocab_size_) {
    vocab_size = vocab_size_;
    W_map.resize(vocab_size * STATE_DIM);
    b_map.resize(vocab_size);
    std::mt19937 rng(999);
    std::normal_distribution<float> wd(0.0f, 0.02f);
    for (int i = 0; i < vocab_size; i++) {
        b_map[i] = 0.0f;
        for (int j = 0; j < STATE_DIM; j++) {
            W_map[i * STATE_DIM + j] = wd(rng);
        }
    }
    selected_word_idx = 0;
    selected_score = 0.0f;
}

int StateToWordMapper::map_to_word(const std::vector<float>& thought_vec,
                                     float valence, float arousal,
                                     const std::vector<float>& embeddings,
                                     int vocab_size_, uint64_t step) const {
    (void)step;
    if (thought_vec.size() < 5 || vocab_size_ <= 0) return 0;

    std::vector<float> state(STATE_DIM, 0.0f);
    for (int i = 0; i < 5 && i < (int)thought_vec.size(); i++)
        state[i] = thought_vec[i];
    state[5] = valence;
    state[6] = arousal;
    state[7] = valence - prev_valence;

    float best_score = -1e10f;
    int best_idx = 0;

    for (int w = 0; w < vocab_size_ && w < vocab_size; w++) {
        float score = b_map[w];
        for (int j = 0; j < STATE_DIM; j++) {
            float sw = W_map[w * STATE_DIM + j];
            float sv = state[j];
            if (!is_bad(sw) && !is_bad(sv)) score += sw * sv;
        }

        if (!embeddings.empty()) {
            float emb_dot = 0.0f;
            int emb_dim = (int)embeddings.size() / vocab_size_;
            for (int d = 0; d < emb_dim && d < (int)thought_vec.size(); d++) {
                float emb = embeddings[w * emb_dim + d];
                float tv = thought_vec[d];
                if (!is_bad(emb) && !is_bad(tv)) emb_dot += emb * tv;
            }
            score += emb_dot * 0.2f;
        }

        if (score > best_score) {
            best_score = score;
            best_idx = w;
        }
    }

    selected_word_idx = best_idx;
    selected_score = best_score;
    prev_valence = valence;
    prev_arousal = arousal;

    return best_idx;
}

void StateToWordMapper::learn(int chosen_idx, float reward) {
    if (chosen_idx < 0 || chosen_idx >= vocab_size) return;
    float lr = 0.005f * std::abs(reward);
    for (int j = 0; j < STATE_DIM; j++) {
        float& w = W_map[chosen_idx * STATE_DIM + j];
        w += lr * (reward > 0 ? 0.1f : -0.05f);
        w = std::max(-0.3f, std::min(0.3f, w));
    }
}