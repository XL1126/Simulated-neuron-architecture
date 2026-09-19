#pragma once
#include <vector>
#include <string>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>

enum class MetaToken {
    NONE,
    UNCERTAIN,
    CORRECTING,
    RETHINKING,
    CONFIDENT
};

struct ConfidenceReport {
    float output_confidence;
    float error_probability;
    float belief_entropy;
    MetaToken suggested_token;
    bool should_gate;
    bool should_rethink;
    float counterfactual_divergence;
};

class MetacognitionMonitor {
public:
    static constexpr int CONFIDENCE_HISTORY = 50;
    static constexpr int ERROR_HISTORY = 20;
    static constexpr float GATE_THRESHOLD = 0.35f;
    static constexpr float RETHINK_THRESHOLD = 0.25f;
    static constexpr float CF_DIVERGENCE_THRESHOLD = 0.5f;

    std::deque<float> confidence_history;
    std::deque<float> error_history;
    std::deque<float> surprise_history;
    std::deque<float> divergence_history;

    float output_confidence;
    float belief_entropy;
    float error_probability;
    float meta_confidence;
    float prev_confidence;

    MetaToken current_token;
    int gated_steps;
    int total_corrections;
    int total_rethinks;

    std::vector<float> output_distribution;
    std::vector<float> predicted_distribution;

    bool is_gated;
    bool is_rethinking;

    std::deque<std::string> meta_token_history;

    MetacognitionMonitor();

    void init();

    ConfidenceReport evaluate(
        const std::vector<float>& output_probs,
        const std::vector<float>& predicted_outcome,
        float world_error, float self_error,
        float meta_surprise, float meta_conf,
        float cf_divergence);

    MetaToken determine_token(const ConfidenceReport& report);

    bool should_gate_output(float confidence, float error_prob);

    bool should_trigger_rethink(float confidence, float cf_divergence);

    void record_outcome(bool was_correct, float actual_error);

    float get_output_confidence() const { return output_confidence; }
    float get_error_probability() const { return error_probability; }
    float get_meta_confidence() const { return meta_confidence; }
    bool get_is_gated() const { return is_gated; }
    bool get_is_rethinking() const { return is_rethinking; }
    MetaToken get_token() const { return current_token; }

    const char* token_to_string(MetaToken t) const;
    std::string get_meta_prefix() const;

    float get_correction_rate() const {
        return total_corrections > 0
            ? (float)(total_corrections) / (float)(total_corrections + total_rethinks + 0.001f)
            : 0.0f;
    }
};