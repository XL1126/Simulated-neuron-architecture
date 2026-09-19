#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>

struct MemoryRecallTester {
    enum TestMode {
        CUED_RECALL,
        FREE_RECALL,
        RECOGNITION,
        TEMPORAL_ORDER
    };

    struct RecallResult {
        bool success;
        float confidence;
        int steps_since_encoding;
        float reconstruction_error;
        std::vector<float> recalled_pattern;
    };

    static constexpr float ERROR_THRESHOLD = 0.5f;

    MemoryRecallTester() = default;

    RecallResult cued_recall(const std::vector<float>& partial_cue,
                              const std::vector<float>& context_hint,
                              int original_encoding_step, int current_step,
                              const std::vector<std::vector<float>>& dg_weights,
                              const std::vector<std::vector<float>>& ca3_weights,
                              const std::vector<std::vector<float>>& ca1_weights,
                              const std::vector<std::vector<float>>& ca1_to_cortex,
                              const std::vector<float>& original_pattern);

    std::vector<RecallResult> adversarial_test(
        const std::vector<int>& test_steps, int max_steps_elapsed,
        int current_step,
        const std::vector<std::vector<float>>& dg_weights,
        const std::vector<std::vector<float>>& ca3_weights,
        const std::vector<std::vector<float>>& ca1_weights,
        const std::vector<std::vector<float>>& ca1_to_cortex,
        const std::vector<std::vector<float>>& original_patterns);

private:
    std::vector<float> sparse_encode_dg(const std::vector<float>& input,
                                         const std::vector<std::vector<float>>& dg_weights);
    std::vector<float> auto_associate_ca3(const std::vector<float>& input,
                                           const std::vector<std::vector<float>>& ca3_weights,
                                           int max_iterations);
    float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    std::vector<float> generate_partial_cue(const std::vector<float>& full_pattern,
                                              float fraction);
};