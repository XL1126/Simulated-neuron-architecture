#include "metacognition_monitor.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

MetacognitionMonitor::MetacognitionMonitor()
    : output_confidence(0.5f), belief_entropy(0.5f), error_probability(0.3f),
      meta_confidence(0.5f), prev_confidence(0.5f),
      current_token(MetaToken::NONE), gated_steps(0),
      total_corrections(0), total_rethinks(0),
      is_gated(false), is_rethinking(false)
{}

void MetacognitionMonitor::init() {
    confidence_history.clear();
    error_history.clear();
    surprise_history.clear();
    divergence_history.clear();
    output_confidence = 0.5f;
    belief_entropy = 0.5f;
    error_probability = 0.3f;
    meta_confidence = 0.5f;
    current_token = MetaToken::NONE;
    gated_steps = 0;
    total_corrections = 0;
    total_rethinks = 0;
    is_gated = false;
    is_rethinking = false;
    output_distribution.clear();
    predicted_distribution.clear();
}

ConfidenceReport MetacognitionMonitor::evaluate(
    const std::vector<float>& output_probs,
    const std::vector<float>& predicted_outcome,
    float world_error, float self_error,
    float meta_surprise, float meta_conf,
    float cf_divergence)
{
    ConfidenceReport report;

    float raw_confidence = 0.5f;
    if (!output_probs.empty()) {
        float max_prob = 0.0f;
        float second_max = 0.0f;
        float ent = 0.0f;
        for (auto p : output_probs) {
            if (is_bad(p)) continue;
            if (p > max_prob) {
                second_max = max_prob;
                max_prob = p;
            } else if (p > second_max) {
                second_max = p;
            }
            if (p > 0.001f) ent -= p * std::log(p);
        }
        raw_confidence = max_prob;
        belief_entropy = std::min(1.0f, ent / 2.0f);
        report.belief_entropy = belief_entropy;
        raw_confidence = raw_confidence * 0.6f + (max_prob - second_max) * 0.4f;
    }

    float outcome_predictability = 0.5f;
    if (!output_probs.empty() && !predicted_outcome.empty()) {
        float kl = 0.0f;
        size_t n = std::min(output_probs.size(), predicted_outcome.size());
        for (size_t i = 0; i < n; i++) {
            float p = std::max(0.001f, output_probs[i]);
            float q = std::max(0.001f, predicted_outcome[i]);
            kl += p * std::log(p / q);
        }
        kl /= (float)n;
        outcome_predictability = 1.0f / (1.0f + kl);
    }

    float error_based_confidence = 1.0f - std::min(1.0f, (world_error + self_error) * 1.5f);

    output_confidence = raw_confidence * 0.35f
        + outcome_predictability * 0.25f
        + error_based_confidence * 0.20f
        + meta_conf * 0.20f;

    if (is_bad(output_confidence)) output_confidence = 0.5f;
    output_confidence = std::max(0.05f, std::min(1.0f, output_confidence));

    error_probability = 1.0f - output_confidence;
    error_probability = error_probability * 0.7f + meta_surprise * 0.3f;

    confidence_history.push_back(output_confidence);
    if ((int)confidence_history.size() > CONFIDENCE_HISTORY)
        confidence_history.pop_front();

    surprise_history.push_back(meta_surprise);
    if ((int)surprise_history.size() > CONFIDENCE_HISTORY)
        surprise_history.pop_front();

    divergence_history.push_back(cf_divergence);
    if ((int)divergence_history.size() > CONFIDENCE_HISTORY)
        divergence_history.pop_front();

    float conf_variance = 0.0f;
    if (confidence_history.size() >= 10) {
        float mean = 0.0f;
        for (auto c : confidence_history) mean += c;
        mean /= (float)confidence_history.size();
        for (auto c : confidence_history) {
            float d = c - mean;
            conf_variance += d * d;
        }
        conf_variance /= (float)confidence_history.size();
    }
    meta_confidence = 1.0f / (1.0f + conf_variance * 8.0f);

    report.output_confidence = output_confidence;
    report.error_probability = error_probability;
    report.counterfactual_divergence = cf_divergence;

    report.should_gate = should_gate_output(output_confidence, error_probability);
    report.should_rethink = should_trigger_rethink(output_confidence, cf_divergence);

    if (report.should_rethink) {
        report.suggested_token = MetaToken::RETHINKING;
    } else if (report.should_gate) {
        report.suggested_token = MetaToken::UNCERTAIN;
    } else if (output_confidence > 0.8f) {
        report.suggested_token = MetaToken::CONFIDENT;
    } else {
        report.suggested_token = MetaToken::NONE;
    }

    prev_confidence = output_confidence;
    return report;
}

MetaToken MetacognitionMonitor::determine_token(const ConfidenceReport& report) {
    if (report.should_rethink) return MetaToken::RETHINKING;
    if (report.should_gate) return MetaToken::UNCERTAIN;
    if (report.output_confidence > 0.85f) return MetaToken::CONFIDENT;
    return MetaToken::NONE;
}

bool MetacognitionMonitor::should_gate_output(float confidence, float error_prob) {
    if (is_gated) {
        gated_steps++;
        if (confidence > GATE_THRESHOLD + 0.15f || gated_steps > 8) {
            is_gated = false;
            gated_steps = 0;
            return false;
        }
        return true;
    }

    if (confidence < GATE_THRESHOLD && error_prob > 0.5f) {
        is_gated = true;
        gated_steps = 1;
        if (current_token == MetaToken::NONE) {
            current_token = MetaToken::UNCERTAIN;
        }
        return true;
    }

    return false;
}

bool MetacognitionMonitor::should_trigger_rethink(float confidence, float cf_divergence) {
    if (is_rethinking) {
        total_rethinks++;
        is_rethinking = false;
        return false;
    }

    if (confidence < RETHINK_THRESHOLD
        && cf_divergence > CF_DIVERGENCE_THRESHOLD
        && !is_gated) {
        is_rethinking = true;
        current_token = MetaToken::RETHINKING;
        return true;
    }

    return false;
}

void MetacognitionMonitor::record_outcome(bool was_correct, float actual_error) {
    error_history.push_back(actual_error);
    if ((int)error_history.size() > ERROR_HISTORY) error_history.pop_front();

    if (!was_correct && current_token == MetaToken::NONE) {
        current_token = MetaToken::CORRECTING;
        meta_token_history.push_back("CORRECTING");
        if (meta_token_history.size() > 20) meta_token_history.pop_front();
        total_corrections++;
    }

    if (was_correct) {
        current_token = MetaToken::NONE;
    }
}

const char* MetacognitionMonitor::token_to_string(MetaToken t) const {
    switch (t) {
        case MetaToken::UNCERTAIN:  return "<uncertain>";
        case MetaToken::CORRECTING: return "<correcting>";
        case MetaToken::RETHINKING: return "<rethinking>";
        case MetaToken::CONFIDENT:  return "<confident>";
        default: return "";
    }
}

std::string MetacognitionMonitor::get_meta_prefix() const {
    if (current_token == MetaToken::NONE) return "";
    if (is_gated) return std::string(token_to_string(MetaToken::UNCERTAIN)) + " ";
    if (is_rethinking) return std::string(token_to_string(MetaToken::RETHINKING)) + " ";
    if (current_token == MetaToken::CORRECTING)
        return std::string(token_to_string(MetaToken::CORRECTING)) + " ";
    return "";
}