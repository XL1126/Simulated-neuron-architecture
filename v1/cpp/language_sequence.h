#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct LanguageSequenceGenerator {
    static constexpr int MAX_WORDS = 4;
    static constexpr int CONTEXT_DIM = 16;

    mutable std::vector<float> seq_context;
    mutable float energy_level;
    mutable float pattern_freshness;
    mutable int output_variety_count;
    mutable int last_best_idx;
    std::vector<std::pair<int, float>> generate_sequence(
        const std::vector<float>& thought_vec,
        const std::vector<float>& narrative_vec,
        float meta_valence, float meta_arousal, float meta_conf,
        const std::vector<float>& embeddings,
        int vocab_size, uint64_t step) const;

    LanguageSequenceGenerator();
    void init();
};