#include "creative_generator.h"
#include <sstream>

CreativeGenerator::CreativeGenerator()
    : n_ideas(0), best_idea_idx(-1),
      creative_context(IDEA_DIM, 0.0f),
      conceptual_blends(CONCEPT_DIM, 0.0f),
      creativity_level(0.2f), divergent_thinking(0.3f),
      convergent_thinking(0.3f), fluency(0.1f),
      originality(0.1f), step_counter(0)
{
    for (int i = 0; i < MAX_IDEAS; i++) {
        idea_pool[i].idea_vector.assign(IDEA_DIM, 0.0f);
        idea_pool[i].source_concepts.assign(CONCEPT_DIM, 0.0f);
        idea_pool[i].novelty = 0.0f;
        idea_pool[i].usefulness = 0.0f;
        idea_pool[i].surprise = 0.0f;
        idea_pool[i].coherence = 0.0f;
        idea_pool[i].total_quality = 0.0f;
        idea_pool[i].created_at = 0;
        idea_pool[i].generation_mode = 0;
    }
}

void CreativeGenerator::init(uint32_t seed) {
    n_ideas = 0;
    best_idea_idx = -1;
    creative_context.assign(IDEA_DIM, 0.0f);
    conceptual_blends.assign(CONCEPT_DIM, 0.0f);
    creativity_level = 0.2f;
    divergent_thinking = 0.3f;
    convergent_thinking = 0.3f;
    fluency = 0.1f;
    originality = 0.1f;
    step_counter = 0;
    idea_history.clear();
    for (int i = 0; i < MAX_IDEAS; i++) {
        idea_pool[i].novelty = 0.0f;
        idea_pool[i].total_quality = 0.0f;
    }
}

void CreativeGenerator::generate_ideas(
    const std::vector<float>& concept_activities,
    const std::vector<float>& self_state,
    const std::vector<float>& emotion_vector,
    const std::vector<float>& episodic_context,
    const std::vector<float>& temporal_context,
    float spontaneity,
    float dopamine)
{
    step_counter++;
    int n_to_generate = 3 + (int)(spontaneity * 3.0f);
    n_to_generate = std::min(n_to_generate, MAX_IDEAS);
    n_ideas = 0;

    for (int i = 0; i < n_to_generate; i++) {
        auto& idea = idea_pool[i];

        float seed = (float)((i * 1103515245 + (int)step_counter * 25214903917ULL) % 10000) / 10000.0f;
        int mode = (int)(seed * 4) % 4;
        idea.generation_mode = mode;

        idea.idea_vector.assign(IDEA_DIM, 0.0f);
        idea.source_concepts.assign(CONCEPT_DIM, 0.0f);

        if (mode == 0) {
            for (int d = 0; d < IDEA_DIM; d++) {
                float sval = d < (int)self_state.size() ? self_state[d] : 0.0f;
                float cval = d < (int)concept_activities.size() ? concept_activities[d] : 0.0f;
                float emo = d < (int)emotion_vector.size() ? emotion_vector[d] : 0.0f;
                float noise = (float)((d * 7 + i * 13) * 1103515245 % 2000 - 1000) / 1000.0f * 0.25f;

                idea.idea_vector[d] = std::tanh(sval * 0.8f + cval * 0.6f + emo * 0.4f + noise) * 0.5f + 0.5f;
                idea.source_concepts[d % CONCEPT_DIM] += idea.idea_vector[d] * 0.1f;
            }
        } else if (mode == 1) {
            float pivot = seed;
            int pivot1 = (int)(pivot * (int)concept_activities.size());
            int pivot2 = (int)((1.0f - pivot) * (int)concept_activities.size() + (int)(seed * 31));

            std::vector<float> c1(IDEA_DIM, 0.0f), c2(IDEA_DIM, 0.0f);
            for (int d = 0; d < IDEA_DIM; d++) {
                int ci1 = (pivot1 + d * 3) % (int)concept_activities.size();
                int ci2 = (pivot2 + d * 7) % (int)concept_activities.size();
                c1[d] = concept_activities[ci1 % concept_activities.size()];
                c2[d] = concept_activities[ci2 % concept_activities.size()];
            }

            _blend_concepts(c1, c2, idea.idea_vector, 0.35f + spontaneity * 0.3f);

            for (int d = 0; d < CONCEPT_DIM; d++) {
                idea.source_concepts[d] = c1[d % c1.size()] * 0.5f + c2[d % c2.size()] * 0.5f;
            }
        } else if (mode == 2) {
            for (int d = 0; d < IDEA_DIM; d++) {
                float emo = d < (int)emotion_vector.size() ? emotion_vector[d] : 0.0f;
                float temp = d < (int)temporal_context.size() ? temporal_context[d] : 0.0f;
                float epi = d < (int)episodic_context.size() ? episodic_context[d] : 0.0f;

                float inversion = 1.0f - emo;
                float noise = (float)((d * 17 + i * 23) * 1103515245 % 2000 - 1000) / 1000.0f * 0.3f;

                idea.idea_vector[d] = inversion * 0.5f + temp * 0.3f + epi * 0.1f + noise;
                idea.idea_vector[d] = std::max(0.0f, std::min(1.0f, idea.idea_vector[d]));
            }
        } else {
            for (int d = 0; d < IDEA_DIM; d++) {
                float noise = (float)(((d * 7 + i * 31) * 1103515245) % 2000 - 1000) / 1000.0f * 0.4f;
                float off = (float)(i * IDEA_DIM + d) * 0.05f;

                idea.idea_vector[d] = 0.5f + noise * 0.5f + off;
                idea.idea_vector[d] = std::max(0.0f, std::min(1.0f, idea.idea_vector[d]));
            }
        }

        idea.novelty = 0.3f + (mode == 1 ? 0.3f : (mode == 2 ? 0.25f : mode == 3 ? 0.4f : 0.15f));
        idea.usefulness = 0.2f + spontaneity * 0.3f;
        idea.surprise = idea.novelty * 0.7f + spontaneity * 0.3f;
        idea.coherence = (mode == 0 ? 0.4f : mode == 1 ? 0.35f : 0.25f);
        idea.total_quality = 0.0f;
        idea.created_at = step_counter;

        n_ideas++;
    }

    fluency = fluency * 0.9f + (float)n_to_generate / (float)MAX_IDEAS * 0.1f;
}

void CreativeGenerator::evaluate_ideas(
    const std::vector<float>& self_state,
    const std::vector<float>& world_state,
    float meta_confidence)
{
    for (int i = 0; i < n_ideas; i++) {
        auto& idea = idea_pool[i];

        float novelty_from_history = _compute_novelty(idea.idea_vector, idea_history);
        idea.novelty = idea.novelty * 0.6f + novelty_from_history * 0.4f;

        float self_alignment = _cosine_sim(idea.idea_vector, self_state);
        float world_alignment = _cosine_sim(idea.idea_vector, world_state);

        idea.usefulness = (self_alignment * 0.4f + world_alignment * 0.3f
                         + meta_confidence * 0.3f);
        idea.usefulness = std::max(0.0f, std::min(1.0f, idea.usefulness));

        idea.surprise = idea.novelty * 0.5f
            + (1.0f - _cosine_sim(idea.idea_vector, creative_context)) * 0.5f;

        idea.coherence = _cosine_sim(idea.idea_vector, self_state);
        idea.coherence = std::max(0.1f, std::min(0.8f, idea.coherence));

        idea.total_quality = idea.novelty * 0.30f
            + idea.usefulness * 0.25f
            + idea.surprise * 0.20f
            + idea.coherence * 0.15f
            + meta_confidence * 0.10f;
    }

    if (n_ideas > 0) {
        float max_q = 0.0f;
        for (int i = 0; i < n_ideas; i++) {
            max_q = std::max(max_q, idea_pool[i].total_quality);
        }
        originality = originality * 0.9f + (max_q > 0.5f ? 0.1f : 0.0f);
    }

    creativity_level = divergent_thinking * 0.4f
        + fluency * 0.2f + originality * 0.4f;
}

void CreativeGenerator::select_best_idea(float randomness) {
    best_idea_idx = -1;

    if (n_ideas == 0) return;

    float explore_prob = randomness * 0.4f + creativity_level * 0.15f;
    float roll = (float)((step_counter * 1103515245) % 10000) / 10000.0f;

    if (n_ideas > 1 && roll < explore_prob) {
        best_idea_idx = (int)(((uint64_t)(step_counter * 25214903917ULL) % n_ideas));
    } else {
        float best = -1e10f;
        for (int i = 0; i < n_ideas; i++) {
            float score = idea_pool[i].total_quality * 0.6f
                + idea_pool[i].novelty * 0.3f
                + idea_pool[i].surprise * 0.1f;
            if (score > best) { best = score; best_idea_idx = i; }
        }
    }

    if (best_idea_idx >= 0 && best_idea_idx < n_ideas) {
        auto& best_idea = idea_pool[best_idea_idx];

        for (int d = 0; d < IDEA_DIM; d++) {
            creative_context[d] = creative_context[d] * 0.85f
                + best_idea.idea_vector[d] * 0.15f;
        }

        idea_history.push_back(best_idea.idea_vector);
        if ((int)idea_history.size() > BLEND_HISTORY)
            idea_history.pop_front();
    }
}

std::vector<float> CreativeGenerator::get_best_idea_vector() const {
    if (best_idea_idx < 0 || best_idea_idx >= n_ideas)
        return std::vector<float>(IDEA_DIM, 0.0f);
    return idea_pool[best_idea_idx].idea_vector;
}

std::string CreativeGenerator::get_idea_description() const {
    if (best_idea_idx < 0 || best_idea_idx >= n_ideas)
        return "no_idea";

    const auto& idea = idea_pool[best_idea_idx];
    std::ostringstream oss;

    const char* modes[] = {"assoc", "blend", "invert", "random"};
    oss << modes[idea.generation_mode % 4] << "_";

    if (idea.novelty > 0.6f) oss << "novel_";
    else if (idea.novelty > 0.3f) oss << "fresh_";
    else oss << "familiar_";

    if (idea.usefulness > 0.5f) oss << "useful";
    else oss << "exploratory";

    return oss.str();
}

std::vector<float> CreativeGenerator::get_creative_modulation(
    const std::vector<float>& concept_activities,
    int n_concepts) const
{
    std::vector<float> mod(n_concepts, 0.0f);

    if (best_idea_idx < 0 || best_idea_idx >= n_ideas)
        return mod;

    const auto& idea = idea_pool[best_idea_idx];

    for (int ci = 0; ci < n_concepts; ci++) {
        float phase = (float)ci / (float)n_concepts * 6.283f;

        float idea_influence = 0.0f;
        for (int d = 0; d < IDEA_DIM; d++) {
            float w = std::sin(phase + (float)d * 0.7f) * 0.5f + 0.5f;
            idea_influence += idea.idea_vector[d] * w * 0.1f;
        }

        mod[ci] = idea_influence * creativity_level * (0.5f + idea.novelty * 0.5f);
        mod[ci] = std::max(-0.3f, std::min(0.3f, mod[ci]));
    }

    return mod;
}

std::vector<float> CreativeGenerator::get_novel_concept_combination(
    int n_concepts) const
{
    std::vector<float> combo(n_concepts, 0.0f);

    if (n_ideas < 2) return combo;

    int i1 = 0;
    int i2 = std::min(1, n_ideas - 1);

    float max_novelty_diff = -1e10f;
    for (int i = 0; i < n_ideas; i++) {
        for (int j = i + 1; j < n_ideas; j++) {
            float diff = std::abs(idea_pool[i].novelty - idea_pool[j].novelty);
            if (diff > max_novelty_diff) {
                max_novelty_diff = diff;
                i1 = i; i2 = j;
            }
        }
    }

    for (int ci = 0; ci < n_concepts; ci++) {
        float phase = (float)ci / (float)n_concepts * 6.283f;

        float v1 = 0.0f, v2 = 0.0f;
        for (int d = 0; d < IDEA_DIM; d++) {
            float w = std::cos(phase + (float)d * 0.3f) * 0.5f + 0.5f;
            v1 += idea_pool[i1].idea_vector[d] * w * 0.15f;
            v2 += idea_pool[i2].idea_vector[d] * w * 0.15f;
        }

        combo[ci] = v1 * 0.5f + v2 * 0.5f;
        combo[ci] = std::max(0.0f, std::min(1.0f, combo[ci]));
    }

    return combo;
}

std::string CreativeGenerator::get_creativity_summary() const {
    std::ostringstream oss;
    oss << "d" << (int)(divergent_thinking * 100)
        << "c" << (int)(convergent_thinking * 100)
        << "f" << (int)(fluency * 100)
        << "o" << (int)(originality * 100);
    return oss.str();
}

void CreativeGenerator::_blend_concepts(
    const std::vector<float>& c1,
    const std::vector<float>& c2,
    std::vector<float>& result,
    float blend_ratio)
{
    for (int d = 0; d < IDEA_DIM; d++) {
        float v1 = d < (int)c1.size() ? c1[d] : 0.0f;
        float v2 = d < (int)c2.size() ? c2[d] : 0.0f;

        float blended = v1 * (1.0f - blend_ratio) + v2 * blend_ratio;
        float emergent = v1 * v2 * 2.0f;

        result[d] = blended * 0.6f + emergent * 0.4f;
        result[d] = std::max(0.0f, std::min(1.0f, result[d]));
    }
}

float CreativeGenerator::_compute_novelty(
    const std::vector<float>& idea,
    const std::deque<std::vector<float>>& history) const
{
    if (history.empty()) return 0.5f;

    float max_sim = -1.0f;
    for (const auto& hist_idea : history) {
        float sim = _cosine_sim(idea, hist_idea);
        if (sim > max_sim) max_sim = sim;
    }

    return 1.0f - std::max(0.0f, max_sim);
}

float CreativeGenerator::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        if (!is_bad(a[i]) && !is_bad(b[i])) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
    }
    float denom = std::sqrt(std::max(1e-10f, na * nb));
    return dot / denom;
}