#include "spontaneous_thinker.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

SpontaneousThinker::SpontaneousThinker()
    : thought_state(THOUGHT_DIM, 0.0f),
      prev_thought_state(THOUGHT_DIM, 0.0f),
      memory_probe(THOUGHT_DIM, 0.0f),
      n_patterns(0), spontaneity(0.5f),
      thought_energy(0.1f), drift_velocity(0.02f),
      current_attractor(-1), dwell_remaining(0),
      is_dwelling(false), step_counter(0),
      base_noise_level(0.03f)
{
    for (int i = 0; i < MAX_PATTERNS; i++) {
        stored_patterns[i].state.resize(THOUGHT_DIM, 0.0f);
        stored_patterns[i].significance = 0.0f;
        stored_patterns[i].source_timestamp = 0;
        stored_patterns[i].recall_count = 0;
        stored_patterns[i].dwell_counter = 0;
    }
}

void SpontaneousThinker::init(uint32_t seed) {
    rng.seed(seed);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int d = 0; d < THOUGHT_DIM; d++) {
        thought_state[d] = dist(rng);
        prev_thought_state[d] = thought_state[d];
    }
    _normalize(thought_state);
    prev_thought_state = thought_state;
    memory_probe.assign(THOUGHT_DIM, 0.0f);
    n_patterns = 0;
    spontaneity = 0.5f;
    thought_energy = 0.1f;
    drift_velocity = 0.02f;
    current_attractor = -1;
    dwell_remaining = 0;
    is_dwelling = false;
    step_counter = 0;
    energy_history.clear();
    base_noise_level = 0.03f;
}

void SpontaneousThinker::think(
    const std::vector<float>& self_state,
    const std::vector<float>& episodic_context,
    float sensory_magnitude,
    float dopamine,
    float norepinephrine,
    float surprise)
{
    step_counter++;

    prev_thought_state = thought_state;

    float ne_noise = norepinephrine * 0.12f;
    float da_boost = dopamine * 0.06f;
    float noise = base_noise_level + ne_noise + (1.0f - sensory_magnitude) * 0.04f;

    for (int d = 0; d < THOUGHT_DIM && d < (int)self_state.size(); d++) {
        float sval = is_bad(self_state[d]) ? 0.0f : self_state[d];
        thought_state[d] = thought_state[d] * 0.85f + sval * 0.15f;
    }

    for (int d = 0; d < THOUGHT_DIM && d < (int)episodic_context.size(); d++) {
        float eval = is_bad(episodic_context[d]) ? 0.0f : episodic_context[d];
        thought_state[d] += eval * 0.05f;
    }

    if (is_dwelling) {
        noise *= 0.3f;
        drift_velocity *= 0.7f;
        dwell_remaining--;
        if (dwell_remaining <= 0) {
            is_dwelling = false;
            current_attractor = -1;
        }
    }

    _attractor_dynamics(noise);

    float energy = 0.0f;
    for (int d = 0; d < THOUGHT_DIM; d++) {
        float diff = thought_state[d] - prev_thought_state[d];
        energy += std::abs(diff);
    }
    thought_energy = energy / (float)THOUGHT_DIM;

    energy_history.push_back(thought_energy);
    if ((int)energy_history.size() > 50) energy_history.pop_front();

    drift_velocity = drift_velocity * 0.9f + thought_energy * 0.1f;

    int closest = _find_closest_pattern();
    if (closest >= 0 && !is_dwelling) {
        float sim = _cosine_sim(thought_state, stored_patterns[closest].state);
        float threshold = 0.55f - dopamine * 0.15f;
        if (sim > threshold) {
            current_attractor = closest;
            dwell_remaining = DWELL_MIN + (int)((float)(rng() % (DWELL_MAX - DWELL_MIN)));
            is_dwelling = true;
            stored_patterns[closest].recall_count++;
            stored_patterns[closest].dwell_counter++;
            float blend = 0.15f;
            for (int d = 0; d < THOUGHT_DIM; d++) {
                thought_state[d] = thought_state[d] * (1.0f - blend)
                    + stored_patterns[closest].state[d] * blend;
            }
            _normalize(thought_state);

            memory_probe = thought_state;
            for (int d = 0; d < THOUGHT_DIM; d++) {
                memory_probe[d] += stored_patterns[closest].state[d] * 0.3f;
            }
            _normalize(memory_probe);
        }
    }

    float inertia = sensory_magnitude * 0.7f + 0.3f;
    spontaneity = spontaneity * (1.0f - 0.01f * inertia)
        + (1.0f - sensory_magnitude) * 0.01f + surprise * 0.02f;
    if (spontaneity > 1.0f) spontaneity = 1.0f;
    if (spontaneity < 0.05f) spontaneity = 0.05f;

    if (!is_dwelling) {
        memory_probe.assign(THOUGHT_DIM, 0.0f);
    }

    _normalize(thought_state);
}

bool SpontaneousThinker::should_trigger_memory_recall() {
    if (!is_dwelling) return false;
    if (current_attractor < 0 || current_attractor >= n_patterns) return false;
    float sig = stored_patterns[current_attractor].significance;
    return sig > 0.3f && dwell_remaining < DWELL_MAX * 0.5f;
}

std::vector<float> SpontaneousThinker::get_memory_probe() const {
    return memory_probe;
}

void SpontaneousThinker::store_pattern(
    const std::vector<float>& experience_summary,
    float significance,
    int timestamp)
{
    if (significance < 0.15f) return;

    int slot = -1;
    if (n_patterns < MAX_PATTERNS) {
        slot = n_patterns;
        n_patterns++;
    } else {
        float min_sig = 1e10f;
        int oldest_idx = 0;
        int oldest_ts = 0x7fffffff;
        for (int i = 0; i < MAX_PATTERNS; i++) {
            float age_penalty = (float)(timestamp - stored_patterns[i].source_timestamp) / 1000.0f;
            float score = stored_patterns[i].significance * 0.4f
                + stored_patterns[i].recall_count * 0.2f - age_penalty * 0.4f;
            if (score < min_sig) { min_sig = score; slot = i; }
            if (stored_patterns[i].source_timestamp < oldest_ts) {
                oldest_ts = stored_patterns[i].source_timestamp;
                oldest_idx = i;
            }
        }
        if (slot < 0) slot = oldest_idx;
    }

    stored_patterns[slot].state.assign(THOUGHT_DIM, 0.0f);
    size_t n = std::min(experience_summary.size(), (size_t)THOUGHT_DIM);
    for (size_t d = 0; d < n; d++) {
        stored_patterns[slot].state[d] = is_bad(experience_summary[d])
            ? 0.0f : experience_summary[d];
    }
    for (size_t d = n; d < THOUGHT_DIM; d++) {
        stored_patterns[slot].state[d] = 0.0f;
    }
    _normalize(stored_patterns[slot].state);
    stored_patterns[slot].significance = significance;
    stored_patterns[slot].source_timestamp = timestamp;
    stored_patterns[slot].recall_count = 0;
    stored_patterns[slot].dwell_counter = 0;
}

std::vector<float> SpontaneousThinker::get_concept_modulation(int n_concepts) const {
    if (n_concepts <= 0) return {};
    std::vector<float> modulation(n_concepts, 0.0f);
    for (int c = 0; c < n_concepts; c++) {
        float val = 0.0f;
        for (int d = 0; d < std::min(THOUGHT_DIM / 2, 8); d++) {
            int idx = (c * 7 + d * 13) % THOUGHT_DIM;
            val += thought_state[idx] * (0.3f + 0.1f * (float)d);
        }
        modulation[c] = std::tanh(val * 2.0f) * 0.5f + 0.5f;
    }
    return modulation;
}

std::vector<float> SpontaneousThinker::get_language_context() const {
    std::vector<float> ctx(16, 0.0f);
    for (int i = 0; i < 16 && i < THOUGHT_DIM; i++) {
        ctx[i] = thought_state[i];
    }
    return ctx;
}

void SpontaneousThinker::_attractor_dynamics(float noise_scale) {
    std::normal_distribution<float> noise_dist(0.0f, noise_scale);
    std::uniform_real_distribution<float> uni_dist(-1.0f, 1.0f);

    for (int iter = 0; iter < ATTRACTOR_ITERS; iter++) {
        std::vector<float> updated(THOUGHT_DIM, 0.0f);

        for (int i = 0; i < std::min(n_patterns, 8); i++) {
            float sim = _cosine_sim(thought_state, stored_patterns[i].state);
            float attract = sim * sim * sim * 0.15f;
            for (int d = 0; d < THOUGHT_DIM; d++) {
                updated[d] += stored_patterns[i].state[d] * attract;
            }
        }

        float self_feedback = 0.0f;
        for (int d = 0; d < THOUGHT_DIM; d++) {
            self_feedback += thought_state[d] * thought_state[d];
        }
        self_feedback = std::sqrt(self_feedback / (float)THOUGHT_DIM);

        for (int d = 0; d < THOUGHT_DIM; d++) {
            thought_state[d] = thought_state[d] * 0.7f
                + updated[d]
                + noise_dist(rng)
                + uni_dist(rng) * self_feedback * 0.01f;
        }

        _normalize(thought_state);
    }
}

void SpontaneousThinker::_compute_energies() {
}

int SpontaneousThinker::_find_closest_pattern() {
    if (n_patterns == 0) return -1;
    float best_sim = -1.0f;
    int best_idx = -1;
    for (int i = 0; i < n_patterns; i++) {
        float sim = _cosine_sim(thought_state, stored_patterns[i].state);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    return best_idx;
}

float SpontaneousThinker::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < THOUGHT_DIM; i++) {
        float av = i < (int)a.size() ? a[i] : 0.0f;
        float bv = i < (int)b.size() ? b[i] : 0.0f;
        if (!is_bad(av) && !is_bad(bv)) {
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
    }
    float denom = std::sqrt(std::max(1e-10f, na * nb));
    return dot / denom;
}

void SpontaneousThinker::_normalize(std::vector<float>& v) const {
    float norm = 0.0f;
    for (auto x : v) if (!is_bad(x)) norm += x * x;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& x : v) x = x / norm * 0.5f;
    }
}