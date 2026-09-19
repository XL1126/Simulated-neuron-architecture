#pragma once
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>

struct EmotionState {
    float joy;
    float sadness;
    float fear;
    float anger;
    float surprise;
    float disgust;
    float anticipation;
    float trust;
    float arousal;
    float valence;
    float dominance;
};

struct EmotionalMemory {
    std::vector<float> context;
    EmotionState emotion;
    float intensity;
    float decay_rate;
    uint64_t timestamp;
};

class EmotionSystem {
public:
    static constexpr int N_EMOTIONS = 8;
    static constexpr int EMOTIONAL_CONTEXT_DIM = 32;
    static constexpr int MAX_EMOTIONAL_MEMORIES = 64;

    EmotionState current_emotion;
    EmotionalMemory emotional_memories[MAX_EMOTIONAL_MEMORIES];
    int n_emotional_memories;
    std::vector<float> emotion_history_embedding;
    std::vector<float> mood_state;
    float mood_stability;
    float emotional_volatility;
    float emotional_depth;
    float emotional_range;
    uint64_t step_counter;
    std::deque<EmotionState> recent_emotions;

    EmotionSystem();

    void init(uint32_t seed);

    void update(
        float reward,
        float novelty,
        float prediction_error,
        float social_feedback,
        float self_model_error,
        float temporal_contrast,
        const std::vector<float>& concept_activities);

    void trigger_emotion(
        int emotion_idx,
        float intensity,
        const std::vector<float>& context);

    void blend_emotions(float blend_rate);

    void regulate_emotions(float serotonin, float norepinephrine);

    void store_emotional_memory(const std::vector<float>& context, float significance);

    std::vector<float> get_emotional_modulation(int target_dim) const;

    std::vector<float> get_emotion_vector() const;

    std::string get_emotion_label() const;

    int get_dominant_emotion() const;

    float get_emotional_intensity() const;

    std::string get_mood_description() const;

private:
    void _decay_emotions(float rate);
    void _update_mood(float alpha);
    void _update_emotion_history();
    int _find_stalest_emotional_memory() const;
    float _sigmoid(float x) const;
    static inline bool is_bad(float x) { return !std::isfinite(x); }
};