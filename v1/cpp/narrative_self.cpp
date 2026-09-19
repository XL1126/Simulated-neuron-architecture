#include "narrative_self.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

NarrativeSelf::NarrativeSelf()
    : narrative_inertia(0.7f), identity_stability(0.5f),
      recent_surprise_ema(0.3f), steps_alive(0)
{}

void NarrativeSelf::init() {
    narrative.assign(NARRATIVE_DIM, 0.0f);
    baseline_self.assign(NARRATIVE_DIM, 0.0f);
    episodic_history.clear();
    valence_history.clear();
    conf_history.clear();
    narrative_inertia = 0.7f;
    identity_stability = 0.5f;
    recent_surprise_ema = 0.3f;
    steps_alive = 0;
    causal_log.clear();
    prev_stability = 0.5f;
    prev_label = "init";
}

void NarrativeSelf::update(const std::vector<float>& self_state,
                            const std::vector<float>& episodic_summary,
                            float meta_conf, float meta_surp,
                            float meta_valence, float intrinsic_reward) {
    steps_alive++;

    std::vector<float> blended(NARRATIVE_DIM, 0.0f);
    size_t copy_n = std::min(self_state.size(), (size_t)NARRATIVE_DIM);
    for (size_t i = 0; i < copy_n; i++) {
        blended[i] = self_state[i] * 0.6f;
    }

    size_t eps_n = std::min(episodic_summary.size(), (size_t)HISTORY_DIM);
    for (size_t i = 0; i < eps_n && i < (size_t)NARRATIVE_DIM; i++) {
        blended[i] += episodic_summary[i] * 0.25f;
    }

    blended[std::min((size_t)60, copy_n)] += meta_valence * 0.3f;
    blended[std::min((size_t)61, copy_n)] += meta_conf * 0.2f;
    blended[std::min((size_t)62, copy_n)] += meta_surp * 0.2f;

    if (episodic_history.size() >= 4) {
        auto& old = episodic_history[episodic_history.size() - 4];
        for (size_t i = 0; i < std::min(old.size(), (size_t)8); i++) {
            blended[std::min((size_t)70 + i, (size_t)NARRATIVE_DIM - 1)] += old[i] * 0.1f;
        }
    }

    if (baseline_self[0] == 0.0f && steps_alive < 10) {
        baseline_self = blended;
    } else {
        float alpha = 0.005f;
        for (size_t i = 0; i < NARRATIVE_DIM; i++) {
            baseline_self[i] = baseline_self[i] * (1.0f - alpha) + blended[i] * alpha;
        }
    }

    for (size_t i = 0; i < NARRATIVE_DIM; i++) {
        if (narrative[i] == 0.0f && !is_bad(blended[i])) {
            narrative[i] = blended[i];
        } else if (!is_bad(blended[i])) {
            narrative[i] = narrative[i] * narrative_inertia + blended[i] * (1.0f - narrative_inertia);
        }
    }

    float norm = 0.0f;
    for (auto& x : narrative) if (!is_bad(x)) norm += x * x;
    if (norm > 1e-8f) {
        norm = std::sqrt(norm);
        for (auto& x : narrative) x = x / norm * 0.5f;
    }

    episodic_history.push_back(episodic_summary);
    if ((int)episodic_history.size() > MAX_HISTORY) episodic_history.pop_front();
    valence_history.push_back(meta_valence);
    if ((int)valence_history.size() > MAX_HISTORY) valence_history.pop_front();
    conf_history.push_back(meta_conf);
    if ((int)conf_history.size() > MAX_HISTORY) conf_history.pop_front();

    recent_surprise_ema = recent_surprise_ema * 0.9f + meta_surp * 0.1f;

    float drift = 0.0f;
    for (size_t i = 0; i < NARRATIVE_DIM; i++) {
        float diff = narrative[i] - baseline_self[i];
        if (!is_bad(diff)) drift += diff * diff;
    }
    drift = std::sqrt(drift / (float)NARRATIVE_DIM);
    identity_stability = 1.0f / (1.0f + drift * 3.0f);

    if (meta_surp > 0.3f) {
        narrative_inertia = std::max(0.4f, narrative_inertia - 0.01f);
    } else if (intrinsic_reward > 0.02f) {
        narrative_inertia = std::min(0.85f, narrative_inertia + 0.005f);
    } else {
        narrative_inertia += (0.65f - narrative_inertia) * 0.002f;
    }

    if (meta_valence < -0.3f) {
        narrative_inertia = std::max(0.35f, narrative_inertia - 0.015f);
    }

    std::string current_label = describe_identity();
    float stab_delta = identity_stability - prev_stability;
    if (current_label != prev_label || std::abs(stab_delta) > 0.1f) {
        std::string why;
        if (meta_surp > 0.3f) why = "surprise";
        else if (meta_valence < -0.2f) why = "distress";
        else if (intrinsic_reward > 0.02f) why = "learning";
        else if (std::abs(stab_delta) > 0.1f) why = stab_delta > 0 ? "settling" : "wavering";
        else why = "flow";
        causal_log.push_back(current_label + "[" + why + "]");
        if ((int)causal_log.size() > 20) causal_log.pop_front();
    }
    prev_stability = identity_stability;
    prev_label = current_label;
}

std::string NarrativeSelf::describe_identity() const {
    std::string s;
    float n0 = narrative.empty() ? 0.0f : narrative[0];
    float n10 = narrative.size() > 10 ? narrative[10] : 0.0f;

    if (identity_stability > 0.7f) s = "stable";
    else if (identity_stability > 0.4f) s = "shifting";
    else s = "confused";

    if (recent_surprise_ema > 0.5f) s += "_surprised";
    if (n0 > 0.1f) s += "_active";
    if (n10 < -0.05f) s += "_cautious";

    return s;
}

std::string NarrativeSelf::get_causal_narrative() const {
    std::string result;
    for (size_t i = 0; i < causal_log.size(); i++) {
        if (i > 0) result += " -> ";
        result += causal_log[i];
    }
    return result.empty() ? "newborn" : result;
}