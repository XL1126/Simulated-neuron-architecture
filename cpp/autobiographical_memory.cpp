#include "autobiographical_memory.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

AutobiographicalMemory::AutobiographicalMemory()
    : consolidation_progress(0.0f), steps_since_consolidation(0),
      total_entries_ever(0)
{
    rng.seed(42);
}

void AutobiographicalMemory::init() {
    entries.clear();
    consolidated_self.assign(SELF_DIM, 0.0f);
    consolidated_world.assign(STATE_DIM, 0.0f);
    consolidation_progress = 0.0f;
    steps_since_consolidation = 0;
    total_entries_ever = 0;
}

float AutobiographicalMemory::compute_significance(
    const std::vector<float>& state,
    const std::vector<float>& self,
    float outcome, float meta_surp) const {

    float novelty = 0.0f;
    if (!entries.empty()) {
        auto& last = entries.back();
        for (size_t i = 0; i < std::min(state.size(), last.state.size()); i++) {
            float diff = state[i] - last.state[i];
            if (!is_bad(diff)) novelty += diff * diff;
        }
        novelty = std::sqrt(novelty / std::min(state.size(), last.state.size()));
    } else {
        novelty = 0.5f;
    }

    float emotional = std::abs(outcome) * 2.0f;
    float surprise = meta_surp;
    float self_relevance = 0.0f;
    for (size_t i = 0; i < std::min(self.size(), (size_t)32); i++) {
        if (!is_bad(self[i])) self_relevance += std::abs(self[i]);
    }
    self_relevance = std::min(1.0f, self_relevance / 16.0f);

    float significance = novelty * 0.35f + emotional * 0.35f + surprise * 0.15f + self_relevance * 0.15f;
    return std::max(0.0f, std::min(1.0f, significance));
}

void AutobiographicalMemory::store(const std::vector<float>& state,
                                    const std::vector<float>& self,
                                    int action, float outcome,
                                    float significance) {
    AutoBioEntry entry;
    entry.state.resize(std::min(state.size(), (size_t)STATE_DIM));
    for (size_t i = 0; i < entry.state.size(); i++)
        entry.state[i] = is_bad(state[i]) ? 0.0f : state[i];

    entry.self.resize(std::min(self.size(), (size_t)SELF_DIM));
    for (size_t i = 0; i < entry.self.size(); i++)
        entry.self[i] = is_bad(self[i]) ? 0.0f : self[i];

    entry.action = action;
    entry.outcome = outcome;
    entry.significance = significance;
    entry.timestamp = total_entries_ever++;
    entry.last_recalled = 0;
    entry.recall_count = 0;

    entries.push_back(entry);
    if ((int)entries.size() > MAX_ENTRIES) {
        float min_sig = 1e10f;
        int min_idx = 0;
        for (int i = 0; i < (int)entries.size() - 10; i++) {
            float age_penalty = (float)(total_entries_ever - entries[i].timestamp) / 10000.0f;
            float recency_bonus = entries[i].recall_count * 0.1f;
            float score = entries[i].significance - age_penalty * 0.3f + recency_bonus;
            if (score < min_sig) { min_sig = score; min_idx = i; }
        }
        entries.erase(entries.begin() + min_idx);
    }
}

std::vector<AutoBioEntry> AutobiographicalMemory::recall(
    const std::vector<float>& cue_self,
    const std::vector<float>& cue_state,
    int top_k, float min_significance) const {

    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < (int)entries.size(); i++) {
        if (entries[i].significance < min_significance) continue;

        float sim_self = 0.0f;
        size_t n_self = std::min(cue_self.size(), entries[i].self.size());
        for (size_t j = 0; j < n_self; j++) {
            if (!is_bad(cue_self[j]) && !is_bad(entries[i].self[j]))
                sim_self += cue_self[j] * entries[i].self[j];
        }
        if (n_self > 0) sim_self /= (float)n_self;

        float sim_state = 0.0f;
        size_t n_state = std::min(cue_state.size(), entries[i].state.size());
        for (size_t j = 0; j < n_state; j++) {
            if (!is_bad(cue_state[j]) && !is_bad(entries[i].state[j]))
                sim_state += cue_state[j] * entries[i].state[j];
        }
        if (n_state > 0) sim_state /= (float)n_state;

        float recency = 1.0f / (1.0f + (float)(total_entries_ever - entries[i].timestamp) / 500.0f);
        float score = sim_self * 0.5f + sim_state * 0.25f
                      + entries[i].significance * 0.15f + recency * 0.1f;
        scores.push_back({score, i});
    }

    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<AutoBioEntry> result;
    for (int i = 0; i < std::min(top_k, (int)scores.size()); i++) {
        result.push_back(entries[scores[i].second]);
    }
    return result;
}

void AutobiographicalMemory::consolidate() {
    if (entries.empty()) return;

    std::vector<float> self_sum(SELF_DIM, 0.0f);
    std::vector<float> world_sum(STATE_DIM, 0.0f);
    float total_weight = 0.0f;

    for (auto& entry : entries) {
        float weight = entry.significance * (1.0f + entry.recall_count * 0.2f);
        if (weight < 0.01f) continue;

        for (size_t i = 0; i < std::min(entry.self.size(), (size_t)SELF_DIM); i++) {
            if (!is_bad(entry.self[i])) self_sum[i] += entry.self[i] * weight;
        }
        for (size_t i = 0; i < std::min(entry.state.size(), (size_t)STATE_DIM); i++) {
            if (!is_bad(entry.state[i])) world_sum[i] += entry.state[i] * weight;
        }
        total_weight += weight;
    }

    if (total_weight > 0.0f) {
        float alpha = 0.1f;
        for (size_t i = 0; i < SELF_DIM; i++) {
            float val = self_sum[i] / total_weight;
            if (!is_bad(val))
                consolidated_self[i] = consolidated_self[i] * (1.0f - alpha) + val * alpha;
        }
        for (size_t i = 0; i < STATE_DIM; i++) {
            float val = world_sum[i] / total_weight;
            if (!is_bad(val))
                consolidated_world[i] = consolidated_world[i] * (1.0f - alpha) + val * alpha;
        }
    }

    consolidation_progress = std::min(1.0f, consolidation_progress + 0.05f);
}

std::string AutobiographicalMemory::summarize_life() const {
    if (entries.empty()) return "newborn";

    float avg_outcome = 0.0f, avg_sig = 0.0f;
    int high_sig_count = 0;
    for (auto& e : entries) {
        avg_outcome += e.outcome;
        avg_sig += e.significance;
        if (e.significance > 0.5f) high_sig_count++;
    }
    avg_outcome /= (float)entries.size();
    avg_sig /= (float)entries.size();

    std::string life;
    if (avg_outcome > 0.1f) life = "thriving";
    else if (avg_outcome > -0.05f) life = "learning";
    else life = "struggling";

    if (high_sig_count > 10) life += "_eventful";
    if (entries.size() > 500) life += "_experienced";
    else if (entries.size() > 200) life += "_growing";
    else life += "_young";

    return life + "[" + std::to_string(entries.size()) + "mem]";
}