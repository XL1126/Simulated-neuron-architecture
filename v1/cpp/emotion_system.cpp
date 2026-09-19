#include "emotion_system.h"
#include <sstream>
#include <random>

EmotionSystem::EmotionSystem()
    : n_emotional_memories(0),
      emotion_history_embedding(EMOTIONAL_CONTEXT_DIM, 0.0f),
      mood_state(8, 0.0f),
      mood_stability(0.5f), emotional_volatility(0.3f),
      emotional_depth(0.1f), emotional_range(0.1f),
      step_counter(0)
{
    current_emotion.joy = 0.1f;
    current_emotion.sadness = 0.05f;
    current_emotion.fear = 0.1f;
    current_emotion.anger = 0.05f;
    current_emotion.surprise = 0.2f;
    current_emotion.disgust = 0.05f;
    current_emotion.anticipation = 0.15f;
    current_emotion.trust = 0.1f;
    current_emotion.arousal = 0.3f;
    current_emotion.valence = 0.1f;
    current_emotion.dominance = 0.5f;

    for (int i = 0; i < MAX_EMOTIONAL_MEMORIES; i++) {
        emotional_memories[i].context.assign(EMOTIONAL_CONTEXT_DIM, 0.0f);
        emotional_memories[i].emotion = current_emotion;
        emotional_memories[i].intensity = 0.0f;
        emotional_memories[i].decay_rate = 0.01f;
        emotional_memories[i].timestamp = 0;
    }
}

void EmotionSystem::init(uint32_t seed) {
    n_emotional_memories = 0;
    emotion_history_embedding.assign(EMOTIONAL_CONTEXT_DIM, 0.0f);
    mood_state.assign(8, 0.0f);
    mood_stability = 0.5f;
    emotional_volatility = 0.3f;
    emotional_depth = 0.1f;
    emotional_range = 0.1f;
    step_counter = 0;
    recent_emotions.clear();
    current_emotion.joy = 0.1f;
    current_emotion.sadness = 0.05f;
    current_emotion.fear = 0.1f;
    current_emotion.anger = 0.05f;
    current_emotion.surprise = 0.2f;
    current_emotion.disgust = 0.05f;
    current_emotion.anticipation = 0.15f;
    current_emotion.trust = 0.1f;
    current_emotion.arousal = 0.3f;
    current_emotion.valence = 0.1f;
    current_emotion.dominance = 0.5f;
}

void EmotionSystem::update(
    float reward,
    float novelty,
    float prediction_error,
    float social_feedback,
    float self_model_error,
    float temporal_contrast,
    const std::vector<float>& concept_activities)
{
    step_counter++;

    float decay = 0.85f;
    current_emotion.joy *= decay;
    current_emotion.sadness *= decay;
    current_emotion.fear *= decay;
    current_emotion.anger *= decay;
    current_emotion.surprise *= decay;
    current_emotion.disgust *= decay;
    current_emotion.anticipation *= decay;
    current_emotion.trust *= decay;

    float reward_effect = std::tanh(reward * 2.0f);
    if (reward > 0.2f) {
        current_emotion.joy += reward_effect * 0.3f;
        current_emotion.trust += reward * 0.15f;
    } else if (reward < -0.1f) {
        current_emotion.sadness += std::abs(reward) * 0.2f;
        current_emotion.disgust += std::abs(reward) * 0.1f;
    }

    if (prediction_error > 0.4f) {
        current_emotion.surprise += prediction_error * 0.25f;
        if (prediction_error > 0.6f) {
            current_emotion.fear += (prediction_error - 0.4f) * 0.3f;
        }
    }

    if (novelty > 0.5f) {
        current_emotion.anticipation += novelty * 0.2f;
        current_emotion.surprise += novelty * 0.1f;
    }

    if (social_feedback < -0.3f) {
        current_emotion.sadness += std::abs(social_feedback) * 0.2f;
        current_emotion.anger += std::abs(social_feedback) * 0.15f;
    } else if (social_feedback > 0.3f) {
        current_emotion.joy += social_feedback * 0.2f;
        current_emotion.trust += social_feedback * 0.15f;
    }

    if (self_model_error > 0.5f) {
        current_emotion.fear += (self_model_error - 0.4f) * 0.15f;
        current_emotion.surprise += self_model_error * 0.1f;
    }

    if (temporal_contrast > 0.3f) {
        current_emotion.surprise += temporal_contrast * 0.15f;
        if (temporal_contrast > 0.6f) {
            current_emotion.sadness += (temporal_contrast - 0.4f) * 0.2f;
        }
    }

    float concepts_sum = 0.0f;
    for (auto v : concept_activities) concepts_sum += v;
    concepts_sum /= std::max(1.0f, (float)concept_activities.size());
    if (concepts_sum > 0.6f) {
        current_emotion.anticipation += (concepts_sum - 0.5f) * 0.15f;
    }

    current_emotion.joy = std::max(0.0f, std::min(1.0f, current_emotion.joy));
    current_emotion.sadness = std::max(0.0f, std::min(1.0f, current_emotion.sadness));
    current_emotion.fear = std::max(0.0f, std::min(1.0f, current_emotion.fear));
    current_emotion.anger = std::max(0.0f, std::min(1.0f, current_emotion.anger));
    current_emotion.surprise = std::max(0.0f, std::min(1.0f, current_emotion.surprise));
    current_emotion.disgust = std::max(0.0f, std::min(1.0f, current_emotion.disgust));
    current_emotion.anticipation = std::max(0.0f, std::min(1.0f, current_emotion.anticipation));
    current_emotion.trust = std::max(0.0f, std::min(1.0f, current_emotion.trust));

    blend_emotions(0.05f);

    float pos_emotions = current_emotion.joy + current_emotion.trust + current_emotion.anticipation;
    float neg_emotions = current_emotion.sadness + current_emotion.fear
                       + current_emotion.anger + current_emotion.disgust;
    float total = pos_emotions + neg_emotions + 0.001f;

    current_emotion.valence = (pos_emotions - neg_emotions) / total;
    current_emotion.valence = std::max(-1.0f, std::min(1.0f, current_emotion.valence));

    current_emotion.arousal = (current_emotion.surprise + current_emotion.fear
                              + current_emotion.joy + current_emotion.anger) / 4.0f;
    current_emotion.arousal = std::max(0.0f, std::min(1.0f, current_emotion.arousal));

    current_emotion.dominance = (current_emotion.anger + current_emotion.anticipation
                                + current_emotion.trust - current_emotion.fear
                                - current_emotion.sadness) * 0.3f + 0.5f;
    current_emotion.dominance = std::max(0.0f, std::min(1.0f, current_emotion.dominance));

    _update_mood(0.02f);
    _update_emotion_history();

    emotional_range = 0.0f;
    float vals[8] = {
        current_emotion.joy, current_emotion.sadness, current_emotion.fear,
        current_emotion.anger, current_emotion.surprise, current_emotion.disgust,
        current_emotion.anticipation, current_emotion.trust
    };
    float em_mean = 0.0f, em_var = 0.0f;
    for (int i = 0; i < 8; i++) em_mean += vals[i];
    em_mean /= 8.0f;
    for (int i = 0; i < 8; i++) em_var += (vals[i] - em_mean) * (vals[i] - em_mean);
    em_var /= 8.0f;
    emotional_range = std::min(1.0f, std::sqrt(em_var) * 4.0f);

    emotional_depth = emotional_depth * 0.98f + emotional_range * 0.02f;
}

void EmotionSystem::trigger_emotion(
    int emotion_idx, float intensity, const std::vector<float>& context)
{
    if (emotion_idx < 0 || emotion_idx >= N_EMOTIONS) return;
    intensity = std::max(0.0f, std::min(1.0f, intensity));

    float* emotion_ptr = nullptr;
    switch (emotion_idx) {
        case 0: emotion_ptr = &current_emotion.joy; break;
        case 1: emotion_ptr = &current_emotion.sadness; break;
        case 2: emotion_ptr = &current_emotion.fear; break;
        case 3: emotion_ptr = &current_emotion.anger; break;
        case 4: emotion_ptr = &current_emotion.surprise; break;
        case 5: emotion_ptr = &current_emotion.disgust; break;
        case 6: emotion_ptr = &current_emotion.anticipation; break;
        case 7: emotion_ptr = &current_emotion.trust; break;
    }
    if (emotion_ptr) {
        *emotion_ptr = std::min(1.0f, *emotion_ptr + intensity);
    }

    if (intensity > 0.5f && !context.empty()) {
        store_emotional_memory(context, intensity);
    }
}

void EmotionSystem::blend_emotions(float blend_rate) {
    float joy = current_emotion.joy;
    float sadness = current_emotion.sadness;
    float fear = current_emotion.fear;
    float anger = current_emotion.anger;
    float surprise = current_emotion.surprise;
    float disgust = current_emotion.disgust;
    float anticipation = current_emotion.anticipation;
    float trust = current_emotion.trust;

    float complex_joy = joy + (anticipation * trust) * blend_rate;
    float complex_sadness = sadness + (disgust * fear) * blend_rate * 0.5f;
    float complex_fear = fear + (surprise * sadness) * blend_rate * 0.5f;
    float awe = surprise + (fear * joy) * blend_rate * 0.3f;
    float remorse = sadness + (disgust * anger) * blend_rate * 0.3f;
    float hope = anticipation + (trust * joy) * blend_rate * 0.3f;

    current_emotion.joy = std::max(0.0f, std::min(1.0f, complex_joy));
    current_emotion.sadness = std::max(0.0f, std::min(1.0f, complex_sadness));
    current_emotion.fear = std::max(0.0f, std::min(1.0f, complex_fear));
    current_emotion.surprise = std::max(0.0f, std::min(1.0f, awe));
    current_emotion.anticipation = std::max(0.0f, std::min(1.0f, hope));
}

void EmotionSystem::regulate_emotions(float serotonin, float norepinephrine) {
    float reg_strength = serotonin * 0.1f;
    current_emotion.fear = std::max(0.0f, current_emotion.fear - reg_strength * 0.5f);
    current_emotion.anger = std::max(0.0f, current_emotion.anger - reg_strength * 0.4f);

    float arousal_boost = norepinephrine * 0.15f;
    current_emotion.surprise = std::min(1.0f, current_emotion.surprise + arousal_boost * 0.3f);
    current_emotion.anticipation = std::min(1.0f, current_emotion.anticipation + arousal_boost * 0.2f);

    emotional_volatility = emotional_volatility * 0.9f + norepinephrine * 0.1f;
    mood_stability = mood_stability * 0.95f + serotonin * 0.05f;
}

void EmotionSystem::store_emotional_memory(
    const std::vector<float>& context, float significance)
{
    int idx = n_emotional_memories;
    if (n_emotional_memories >= MAX_EMOTIONAL_MEMORIES) {
        idx = _find_stalest_emotional_memory();
    } else {
        n_emotional_memories++;
    }

    auto& mem = emotional_memories[idx];
    for (int d = 0; d < EMOTIONAL_CONTEXT_DIM && d < (int)context.size(); d++) {
        mem.context[d] = context[d];
    }
    mem.emotion = current_emotion;
    mem.intensity = significance;
    mem.decay_rate = 0.005f + (1.0f - significance) * 0.02f;
    mem.timestamp = step_counter;

    for (int d = 0; d < EMOTIONAL_CONTEXT_DIM; d++) {
        emotion_history_embedding[d] = emotion_history_embedding[d] * 0.95f
            + mem.context[d] * 0.05f;
    }
}

std::vector<float> EmotionSystem::get_emotional_modulation(int target_dim) const {
    std::vector<float> mod(target_dim, 0.0f);
    for (int d = 0; d < target_dim; d++) {
        float seed = (float)(d * 7 + 1);
        float phase = seed * 3.14159f;

        mod[d] = current_emotion.joy * (0.5f + 0.5f * std::sin(phase))
               + current_emotion.sadness * (0.3f + 0.3f * std::cos(phase * 2.0f)) * -0.5f
               + current_emotion.fear * std::abs(std::sin(phase * 3.0f)) * -0.4f
               + current_emotion.anger * (0.4f + 0.4f * std::sin(phase * 1.5f + 1.0f))
               + current_emotion.surprise * std::abs(std::cos(phase * 2.5f))
               + current_emotion.anticipation * (0.3f + 0.3f * std::cos(phase * 0.7f))
               + current_emotion.trust * (0.2f + 0.2f * std::sin(phase * 0.5f));

        mod[d] = std::max(-1.0f, std::min(1.0f, mod[d]));
    }
    return mod;
}

std::vector<float> EmotionSystem::get_emotion_vector() const {
    std::vector<float> vec(8);
    vec[0] = current_emotion.joy;
    vec[1] = current_emotion.sadness;
    vec[2] = current_emotion.fear;
    vec[3] = current_emotion.anger;
    vec[4] = current_emotion.surprise;
    vec[5] = current_emotion.disgust;
    vec[6] = current_emotion.anticipation;
    vec[7] = current_emotion.trust;
    return vec;
}

std::string EmotionSystem::get_emotion_label() const {
    int dom = get_dominant_emotion();
    const char* labels[] = {"joy", "sadness", "fear", "anger",
                            "surprise", "disgust", "anticipation", "trust"};
    float intensity = get_emotional_intensity();

    std::ostringstream oss;
    if (intensity < 0.15f) oss << "neutral";
    else if (intensity < 0.35f) oss << "mild_" << labels[dom];
    else if (intensity < 0.6f) oss << "moderate_" << labels[dom];
    else oss << "strong_" << labels[dom];

    if (current_emotion.joy > 0.3f && dom != 0)
        oss << "_with_joy";
    if (current_emotion.fear > 0.3f && dom != 2)
        oss << "_with_fear";
    if (current_emotion.anticipation > 0.3f && dom != 6)
        oss << "_with_anticipation";

    return oss.str();
}

int EmotionSystem::get_dominant_emotion() const {
    float vals[8] = {
        current_emotion.joy, current_emotion.sadness, current_emotion.fear,
        current_emotion.anger, current_emotion.surprise, current_emotion.disgust,
        current_emotion.anticipation, current_emotion.trust
    };
    int best = 4;
    float best_val = vals[4];
    for (int i = 0; i < 8; i++) {
        if (vals[i] > best_val) { best_val = vals[i]; best = i; }
    }
    return best;
}

float EmotionSystem::get_emotional_intensity() const {
    float total = 0.0f;
    float vals[8] = {
        current_emotion.joy, current_emotion.sadness, current_emotion.fear,
        current_emotion.anger, current_emotion.surprise, current_emotion.disgust,
        current_emotion.anticipation, current_emotion.trust
    };
    for (int i = 0; i < 8; i++) total += vals[i];
    return std::min(1.0f, total / 4.0f);
}

std::string EmotionSystem::get_mood_description() const {
    float mood_sum = 0.0f;
    for (auto v : mood_state) mood_sum += v;

    std::ostringstream oss;
    if (mood_stability > 0.7f) oss << "stable_";
    else if (mood_stability > 0.4f) oss << "balanced_";
    else oss << "volatile_";

    if (current_emotion.valence > 0.3f) oss << "positive";
    else if (current_emotion.valence < -0.3f) oss << "negative";
    else oss << "neutral";

    oss << "_d" << (int)(emotional_depth * 100);

    return oss.str();
}

void EmotionSystem::_decay_emotions(float rate) {
    current_emotion.joy *= (1.0f - rate);
    current_emotion.sadness *= (1.0f - rate);
    current_emotion.fear *= (1.0f - rate);
    current_emotion.anger *= (1.0f - rate);
    current_emotion.surprise *= (1.0f - rate);
    current_emotion.disgust *= (1.0f - rate);
    current_emotion.anticipation *= (1.0f - rate);
    current_emotion.trust *= (1.0f - rate);
}

void EmotionSystem::_update_mood(float alpha) {
    float vals[8] = {
        current_emotion.joy, current_emotion.sadness, current_emotion.fear,
        current_emotion.anger, current_emotion.surprise, current_emotion.disgust,
        current_emotion.anticipation, current_emotion.trust
    };
    for (int i = 0; i < 8; i++) {
        mood_state[i] = mood_state[i] * (1.0f - alpha * (1.0f - mood_stability))
                      + vals[i] * alpha * (1.0f - mood_stability);
    }
}

void EmotionSystem::_update_emotion_history() {
    recent_emotions.push_back(current_emotion);
    if ((int)recent_emotions.size() > 50) recent_emotions.pop_front();
}

int EmotionSystem::_find_stalest_emotional_memory() const {
    int stalest = 0;
    uint64_t min_time = emotional_memories[0].timestamp;
    for (int i = 1; i < MAX_EMOTIONAL_MEMORIES; i++) {
        if (emotional_memories[i].timestamp < min_time) {
            min_time = emotional_memories[i].timestamp;
            stalest = i;
        }
    }
    return stalest;
}

float EmotionSystem::_sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(std::max(-10.0f, std::min(10.0f, -x))));
}