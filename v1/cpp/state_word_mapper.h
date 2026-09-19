#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct StateToWordMapper {
    static constexpr int EMBED_DIM = 32;
    static constexpr int STATE_DIM = 8;

    std::vector<float> W_map;
    std::vector<float> b_map;

    int vocab_size;
    mutable int selected_word_idx;
    mutable float selected_score;
    mutable float prev_valence;
    mutable float prev_arousal;

    StateToWordMapper();

    void init(int vocab_size);

    int map_to_word(const std::vector<float>& thought_vec,
                     float valence, float arousal,
                     const std::vector<float>& embeddings,
                     int vocab_size, uint64_t step) const;

    void learn(int chosen_idx, float reward);
};