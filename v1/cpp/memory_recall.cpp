#include "memory_recall.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>

static inline bool is_bad(float x) { return !std::isfinite(x); }

MemoryRecallTester::RecallResult MemoryRecallTester::cued_recall(
    const std::vector<float>& partial_cue,
    const std::vector<float>& context_hint,
    int original_encoding_step, int current_step,
    const std::vector<std::vector<float>>& dg_weights,
    const std::vector<std::vector<float>>& ca3_weights,
    const std::vector<std::vector<float>>& ca1_weights,
    const std::vector<std::vector<float>>& ca1_to_cortex,
    const std::vector<float>& original_pattern)
{
    (void)context_hint;
    RecallResult result;
    result.steps_since_encoding = current_step - original_encoding_step;
    result.success = false;
    result.reconstruction_error = 999.0f;

    if (partial_cue.empty()) return result;

    auto dg_pattern = sparse_encode_dg(partial_cue, dg_weights);
    if (dg_pattern.empty()) return result;

    auto ca3_pattern = auto_associate_ca3(dg_pattern, ca3_weights, 20);
    if (ca3_pattern.empty()) return result;

    std::vector<float> ca1_output(ca1_weights.size(), 0.0f);
    for (size_t i = 0; i < ca1_weights.size() && i < ca3_pattern.size(); i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < ca1_weights[i].size() && j < ca3_pattern.size(); j++) {
            float w = ca1_weights[i][j];
            float ca3 = ca3_pattern[j];
            if (!is_bad(w) && !is_bad(ca3)) sum += w * ca3;
        }
        ca1_output[i] = std::tanh(sum);
        if (is_bad(ca1_output[i])) ca1_output[i] = 0.0f;
    }

    result.recalled_pattern.resize(ca1_to_cortex.empty() ? 0 : ca1_to_cortex[0].size(), 0.0f);
    for (size_t j = 0; j < result.recalled_pattern.size(); j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < ca1_to_cortex.size() && i < ca1_output.size(); i++) {
            if (j < ca1_to_cortex[i].size()) {
                float w = ca1_to_cortex[i][j];
                float ca1 = ca1_output[i];
                if (!is_bad(w) && !is_bad(ca1)) sum += w * ca1;
            }
        }
        result.recalled_pattern[j] = std::tanh(sum);
        if (is_bad(result.recalled_pattern[j])) result.recalled_pattern[j] = 0.0f;
    }

    if (!original_pattern.empty() && !result.recalled_pattern.empty()) {
        result.reconstruction_error = euclidean_distance(
            original_pattern, result.recalled_pattern);
    }

    result.success = (result.reconstruction_error < ERROR_THRESHOLD);
    result.confidence = 1.0f - std::min(result.reconstruction_error / 2.0f, 1.0f);
    if (is_bad(result.confidence)) result.confidence = 0.0f;

    return result;
}

std::vector<MemoryRecallTester::RecallResult> MemoryRecallTester::adversarial_test(
    const std::vector<int>& test_steps, int max_steps_elapsed,
    int current_step,
    const std::vector<std::vector<float>>& dg_weights,
    const std::vector<std::vector<float>>& ca3_weights,
    const std::vector<std::vector<float>>& ca1_weights,
    const std::vector<std::vector<float>>& ca1_to_cortex,
    const std::vector<std::vector<float>>& original_patterns)
{
    std::vector<RecallResult> results;
    for (size_t idx = 0; idx < test_steps.size(); idx++) {
        int orig_step = test_steps[idx];
        if (current_step - orig_step > max_steps_elapsed) continue;
        if (idx >= original_patterns.size()) continue;

        auto partial_cue = generate_partial_cue(original_patterns[idx], 0.5f);
        auto result = cued_recall(partial_cue, {}, orig_step, current_step,
                                   dg_weights, ca3_weights, ca1_weights,
                                   ca1_to_cortex, original_patterns[idx]);
        results.push_back(result);
    }
    return results;
}

std::vector<float> MemoryRecallTester::sparse_encode_dg(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& dg_weights)
{
    if (input.empty() || dg_weights.empty()) return {};

    std::vector<float> output(dg_weights.size(), 0.0f);
    for (size_t i = 0; i < dg_weights.size(); i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < dg_weights[i].size() && j < input.size(); j++) {
            float w = dg_weights[i][j];
            float inp = input[j];
            if (!is_bad(w) && !is_bad(inp)) sum += w * inp;
        }
        output[i] = std::tanh(sum);
        if (is_bad(output[i])) output[i] = 0.0f;
    }

    float max_val = 0.0f;
    for (auto& v : output) { if (v > max_val) max_val = v; }
    float threshold = max_val * 0.9f;
    for (auto& v : output) { if (v < threshold) v = 0.0f; }

    return output;
}

std::vector<float> MemoryRecallTester::auto_associate_ca3(
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& ca3_weights,
    int max_iterations)
{
    if (input.empty() || ca3_weights.empty()) return {};

    size_t sz = ca3_weights.size();
    std::vector<float> pattern = input;

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<float> next(sz, 0.0f);
        for (size_t i = 0; i < sz; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < ca3_weights[i].size() && j < pattern.size(); j++) {
                float w = ca3_weights[i][j];
                float p = pattern[j];
                if (!is_bad(w) && !is_bad(p)) sum += w * p;
            }
            next[i] = std::tanh(sum);
            if (is_bad(next[i])) next[i] = 0.0f;
        }

        float delta = cosine_distance(pattern, next);
        pattern = next;
        if (delta < 0.001f) break;
    }

    return pattern;
}

float MemoryRecallTester::cosine_distance(const std::vector<float>& a,
                                            const std::vector<float>& b) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return 1.0f;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float ai = is_bad(a[i]) ? 0.0f : a[i];
        float bi = is_bad(b[i]) ? 0.0f : b[i];
        dot += ai * bi; na += ai * ai; nb += bi * bi;
    }

    if (na < 0.0001f || nb < 0.0001f) return 1.0f;
    float cos_sim = dot / (std::sqrt(na) * std::sqrt(nb));
    if (is_bad(cos_sim)) return 1.0f;
    return std::max(0.0f, 1.0f - cos_sim);
}

float MemoryRecallTester::euclidean_distance(const std::vector<float>& a,
                                               const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 999.0f;
    size_t sz = std::min(a.size(), b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float diff = (is_bad(a[i]) ? 0.0f : a[i]) - (is_bad(b[i]) ? 0.0f : b[i]);
        sum += diff * diff;
    }
    float result = std::sqrt(sum);
    return is_bad(result) ? 999.0f : result;
}

std::vector<float> MemoryRecallTester::generate_partial_cue(
    const std::vector<float>& full_pattern, float fraction)
{
    if (full_pattern.empty()) return {};
    std::vector<float> partial = full_pattern;
    int n_mask = static_cast<int>(partial.size() * (1.0f - fraction));
    for (int i = 0; i < n_mask; i++) {
        int idx = rand() % static_cast<int>(partial.size());
        partial[idx] = 0.0f;
    }
    return partial;
}