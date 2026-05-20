#include "temporal_depth.h"
#include <random>
#include <sstream>

TemporalDepth::TemporalDepth()
    : temporal_chain(TEMPORAL_DIM, 0.0f),
      prev_temporal_chain(TEMPORAL_DIM, 0.0f),
      n_chapters(0), temporal_coherence(0.5f),
      temporal_depth_score(0.0f), anticipation_accuracy(0.3f),
      nostalgia_level(0.0f), step_counter(0),
      current_chapter_start(0)
{
    for (int i = 0; i < PROJECTION_HORIZON; i++) {
        future_projections[i].projected_self.assign(TEMPORAL_DIM, 0.0f);
        future_projections[i].confidence = 0.3f;
        future_projections[i].timescale = (float)(i + 1);
        future_projections[i].expected_valence = 0.0f;
    }
    for (int i = 0; i < MAX_CHAPTERS; i++) {
        life_chapters[i].theme.assign(TEMPORAL_DIM, 0.0f);
        life_chapters[i].start_time = 0;
        life_chapters[i].end_time = 0;
        life_chapters[i].coherence = 0.0f;
        life_chapters[i].n_events = 0;
    }
}

void TemporalDepth::init(uint32_t seed) {
    temporal_chain.assign(TEMPORAL_DIM, 0.0f);
    prev_temporal_chain.assign(TEMPORAL_DIM, 0.0f);
    time_slices.clear();
    n_chapters = 0;
    temporal_coherence = 0.5f;
    temporal_depth_score = 0.0f;
    anticipation_accuracy = 0.3f;
    nostalgia_level = 0.0f;
    step_counter = 0;
    current_chapter_start = 0;
    for (int i = 0; i < PROJECTION_HORIZON; i++) {
        future_projections[i].projected_self.assign(TEMPORAL_DIM, 0.0f);
        future_projections[i].confidence = 0.3f;
        future_projections[i].timescale = (float)(i + 1);
        future_projections[i].expected_valence = 0.0f;
    }
    for (int i = 0; i < MAX_CHAPTERS; i++) {
        life_chapters[i].theme.assign(TEMPORAL_DIM, 0.0f);
        life_chapters[i].start_time = 0;
        life_chapters[i].end_time = 0;
        life_chapters[i].coherence = 0.0f;
        life_chapters[i].n_events = 0;
    }
}

void TemporalDepth::record_slice(
    const std::vector<float>& self_state,
    const std::vector<float>& episodic_context,
    float significance,
    float valence)
{
    if (significance < 0.05f) return;

    TimeSlice slice;
    slice.self_state.assign(self_state.begin(), self_state.end());
    slice.self_state.resize(TEMPORAL_DIM, 0.0f);
    slice.context.assign(episodic_context.begin(), episodic_context.end());
    slice.context.resize(TEMPORAL_DIM, 0.0f);
    slice.significance = significance;
    slice.valence = valence;
    slice.timestamp = step_counter;

    time_slices.push_back(slice);
    if ((int)time_slices.size() > MAX_SLICES) time_slices.pop_front();
}

void TemporalDepth::update_temporal_chain(
    const std::vector<float>& recent_self,
    const std::vector<float>& autobiographical_summary,
    float meta_conf,
    float world_change_rate)
{
    step_counter++;
    prev_temporal_chain = temporal_chain;

    for (int d = 0; d < TEMPORAL_DIM && d < (int)recent_self.size(); d++) {
        float rval = is_bad(recent_self[d]) ? 0.0f : recent_self[d];
        temporal_chain[d] = temporal_chain[d] * 0.88f + rval * 0.12f;
    }

    for (int d = 0; d < TEMPORAL_DIM && d < (int)autobiographical_summary.size(); d++) {
        float aval = is_bad(autobiographical_summary[d]) ? 0.0f : autobiographical_summary[d];
        temporal_chain[d] += aval * 0.05f;
    }

    float diff = 0.0f;
    for (int d = 0; d < TEMPORAL_DIM; d++) {
        float dval = temporal_chain[d] - prev_temporal_chain[d];
        diff += dval * dval;
    }
    float change_rate = std::sqrt(diff / (float)TEMPORAL_DIM);

    temporal_coherence = temporal_coherence * 0.90f
        + (1.0f - std::min(1.0f, change_rate * 3.0f)) * 0.10f;

    float weighted_past = 0.0f;
    float total_weight = 0.0f;
    int n = (int)time_slices.size();
    for (int i = 0; i < n; i++) {
        float age = (float)(step_counter - time_slices[i].timestamp) / 1000.0f;
        float weight = std::exp(-age * 0.5f) * time_slices[i].significance;
        weighted_past += time_slices[i].valence * weight;
        total_weight += weight;
    }

    nostalgia_level = nostalgia_level * 0.95f;
    if (total_weight > 1e-6f) {
        float avg_past_valence = weighted_past / total_weight;
        float current_valence = 0.0f;
        for (int d = 0; d < TEMPORAL_DIM && d < (int)recent_self.size(); d++) {
            current_valence += recent_self[d];
        }
        current_valence /= (float)TEMPORAL_DIM;
        if (avg_past_valence > current_valence + 0.05f) {
            nostalgia_level += 0.02f;
        }
    }
    nostalgia_level = std::min(1.0f, nostalgia_level);

    temporal_depth_score = temporal_coherence * 0.5f
        + std::min(1.0f, (float)time_slices.size() / 20.0f) * 0.2f
        + anticipation_accuracy * 0.15f
        + (n_chapters > 0 ? 0.15f : 0.0f);
}

void TemporalDepth::project_future(
    const std::vector<float>& current_self,
    const std::vector<float>& world_state,
    const std::vector<float>& trend_vector,
    float dopamine)
{
    for (int h = 0; h < PROJECTION_HORIZON; h++) {
        float scale = (float)(h + 1);
        future_projections[h].timescale = scale;

        for (int d = 0; d < TEMPORAL_DIM; d++) {
            float cur = d < (int)current_self.size() ? current_self[d] : 0.0f;
            float trend = d < (int)trend_vector.size() ? trend_vector[d] : 0.0f;
            float world = d < (int)world_state.size() ? world_state[d] : 0.0f;

            future_projections[h].projected_self[d] =
                cur * (1.0f - 0.15f * scale) + trend * 0.15f * scale + world * 0.05f * scale;

            if (is_bad(future_projections[h].projected_self[d]))
                future_projections[h].projected_self[d] = 0.0f;
        }

        float consistency = _cosine_sim(
            future_projections[h].projected_self, current_self);
        future_projections[h].confidence = 0.4f
            + consistency * 0.3f + temporal_coherence * 0.2f - scale * 0.1f;
        future_projections[h].confidence = std::max(0.1f,
            std::min(1.0f, future_projections[h].confidence));

        float expected_val = 0.0f;
        for (int d = 0; d < TEMPORAL_DIM; d++) {
            expected_val += future_projections[h].projected_self[d];
        }
        future_projections[h].expected_valence =
            expected_val / (float)TEMPORAL_DIM * dopamine;
    }
}

void TemporalDepth::update_chapters(
    const std::vector<float>& current_self,
    float self_change_magnitude,
    float significance_threshold)
{
    if (n_chapters == 0) {
        n_chapters = 1;
        life_chapters[0].start_time = step_counter;
        life_chapters[0].theme.assign(TEMPORAL_DIM, 0.0f);
        for (int d = 0; d < TEMPORAL_DIM && d < (int)current_self.size(); d++)
            life_chapters[0].theme[d] = current_self[d];
        life_chapters[0].coherence = 1.0f;
        life_chapters[0].n_events = 1;
        current_chapter_start = step_counter;
        return;
    }

    auto& current_chapter = life_chapters[n_chapters - 1];
    float sim = _cosine_sim(current_chapter.theme, current_self);

    if (self_change_magnitude > 0.4f && sim < 0.6f && n_chapters < MAX_CHAPTERS) {
        current_chapter.end_time = step_counter;
        n_chapters++;
        auto& new_ch = life_chapters[n_chapters - 1];
        new_ch.start_time = step_counter;
        new_ch.theme.assign(TEMPORAL_DIM, 0.0f);
        for (int d = 0; d < TEMPORAL_DIM && d < (int)current_self.size(); d++)
            new_ch.theme[d] = current_self[d];
        new_ch.coherence = 1.0f;
        new_ch.n_events = 1;
        current_chapter_start = step_counter;
    } else {
        current_chapter.end_time = step_counter;
        for (int d = 0; d < TEMPORAL_DIM && d < (int)current_self.size(); d++) {
            current_chapter.theme[d] = current_chapter.theme[d] * 0.95f
                + current_self[d] * 0.05f;
        }
        current_chapter.n_events++;
        current_chapter.coherence = current_chapter.coherence * 0.9f + sim * 0.1f;
    }
}

float TemporalDepth::evaluate_anticipation(
    const std::vector<float>& actual_self,
    float timestep)
{
    int h = (int)(timestep - 0.5f);
    if (h < 0) h = 0;
    if (h >= PROJECTION_HORIZON) h = PROJECTION_HORIZON - 1;

    float err = 0.0f;
    int count = 0;
    for (int d = 0; d < TEMPORAL_DIM && d < (int)actual_self.size(); d++) {
        float diff = actual_self[d] - future_projections[h].projected_self[d];
        err += diff * diff;
        count++;
    }
    if (count > 0) err = std::sqrt(err / (float)count);

    float accuracy = std::max(0.0f, 1.0f - err * 2.0f);
    anticipation_accuracy = anticipation_accuracy * 0.95f + accuracy * 0.05f;
    return accuracy;
}

std::vector<float> TemporalDepth::get_temporal_context() const {
    return temporal_chain;
}

std::vector<float> TemporalDepth::get_future_shadow(int horizon_idx) const {
    if (horizon_idx < 0 || horizon_idx >= PROJECTION_HORIZON) return {};
    return future_projections[horizon_idx].projected_self;
}

std::string TemporalDepth::get_life_chapter_label() const {
    if (n_chapters == 0) return "beginning";
    auto& ch = life_chapters[n_chapters - 1];
    float theme_sum = 0.0f;
    for (auto v : ch.theme) theme_sum += v;
    theme_sum /= (float)TEMPORAL_DIM;

    if (ch.coherence > 0.8f && theme_sum > 0.5f) return "thriving";
    if (ch.coherence > 0.6f) return "stable_period";
    if (ch.coherence < 0.3f) return "transition";
    if (theme_sum > 0.7f) return "growth";
    if (theme_sum < 0.2f) return "struggling";
    return "learning";
}

std::string TemporalDepth::get_timeline_summary() const {
    std::ostringstream oss;
    oss << (int)time_slices.size() << "slices_";
    oss << n_chapters << "chapters_";
    oss << "depth" << (int)(temporal_depth_score * 100);
    return oss.str();
}

float TemporalDepth::get_temporal_continuity() const {
    return temporal_coherence;
}

std::vector<float> TemporalDepth::get_future_modulation(int dim) const {
    std::vector<float> mod(dim, 0.0f);
    int h = 0;
    float w_sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float val = 0.0f;
        for (int ph = 0; ph < std::min(2, PROJECTION_HORIZON); ph++) {
            float weight = 1.0f / (float)(ph + 1);
            if (d < TEMPORAL_DIM) {
                val += future_projections[ph].projected_self[d] * weight;
                w_sum += weight;
            }
        }
        mod[d] = w_sum > 0 ? val / w_sum : 0.0f;
    }
    return mod;
}

float TemporalDepth::_compute_coherence(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    return _cosine_sim(a, b);
}

float TemporalDepth::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < TEMPORAL_DIM; i++) {
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

void TemporalDepth::_detect_chapter_boundary(
    const std::vector<float>& current, const std::vector<float>& previous)
{
}