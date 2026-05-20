#pragma once
#include <vector>
#include <string>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct AutoBioEntry {
    std::vector<float> state;
    std::vector<float> self;
    int action;
    float outcome;
    float significance;
    uint64_t timestamp;
    uint64_t last_recalled;
    int recall_count;
};

struct AutobiographicalMemory {
    static constexpr int MAX_ENTRIES = 1024;
    static constexpr int STATE_DIM = 64;
    static constexpr int SELF_DIM = 128;
    static constexpr int CONSOLIDATION_INTERVAL = 500;

    std::deque<AutoBioEntry> entries;
    std::vector<float> consolidated_self;
    std::vector<float> consolidated_world;
    float consolidation_progress;
    int steps_since_consolidation;
    uint64_t total_entries_ever;

    std::mt19937 rng;

    AutobiographicalMemory();

    void init();

    void store(const std::vector<float>& state,
               const std::vector<float>& self,
               int action, float outcome,
               float significance);

    std::vector<AutoBioEntry> recall(const std::vector<float>& cue_self,
                                     const std::vector<float>& cue_state,
                                     int top_k, float min_significance) const;

    void consolidate();

    float compute_significance(const std::vector<float>& state,
                                const std::vector<float>& self,
                                float outcome, float meta_surp) const;

    const std::vector<float>& get_consolidated_self() const { return consolidated_self; }

    std::string summarize_life() const;
};