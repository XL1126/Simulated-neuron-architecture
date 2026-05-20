#include "replay_engine.h"
#include <algorithm>
#include <cstdlib>
#include <random>

static inline bool is_bad(float x) { return !std::isfinite(x); }

static std::mt19937 replay_rng(static_cast<unsigned>(time(nullptr)));

ReplayEngine::ReplayEngine()
    : buffer_write_pos(0), buffer_count(0), replay_pos(0), replay_active(false)
{
    ca3_buffer.resize(BUFFER_SIZE);
}

void ReplayEngine::init(size_t ca3_size) {
    ca3_buffer.clear();
    ca3_buffer.resize(BUFFER_SIZE, std::vector<float>(ca3_size, 0.0f));
    buffer_write_pos = 0;
    buffer_count = 0;
    replay_sequence.clear();
    replay_pos = 0;
    replay_active = false;
}

void ReplayEngine::record_ca3_snapshot(const std::vector<float>& ca3_activity) {
    if (ca3_activity.empty() || buffer_write_pos >= BUFFER_SIZE) return;
    ca3_buffer[buffer_write_pos] = ca3_activity;
    buffer_write_pos = (buffer_write_pos + 1) % BUFFER_SIZE;
    buffer_count = std::min(buffer_count + 1, BUFFER_SIZE);
}

void ReplayEngine::init_replay_sequence(size_t ca3_size) {
    replay_sequence.clear();
    replay_pos = 0;

    if (buffer_count < 3 || ca3_size == 0) return;

    std::uniform_int_distribution<int> offset_dist(1, std::min(buffer_count, 20));
    int recent_offset = offset_dist(replay_rng);
    int seed_idx = (buffer_write_pos - recent_offset + BUFFER_SIZE) % BUFFER_SIZE;

    std::vector<float> current = ca3_buffer[seed_idx];
    if (current.empty() || current.size() != ca3_size) return;

    int replay_len = 10 + (replay_rng() % 15);
    std::normal_distribution<float> noise(0.0f, 0.03f);
    int buf_idx = (seed_idx + 1) % BUFFER_SIZE;

    for (int i = 0; i < replay_len; i++) {
        std::vector<float> frame = current;
        for (auto& v : frame) {
            if (is_bad(v)) v = 0.0f;
            v += noise(replay_rng);
        }
        replay_sequence.push_back(frame);

        if (buf_idx != seed_idx &&
            !ca3_buffer[buf_idx].empty() &&
            ca3_buffer[buf_idx].size() == ca3_size) {
            current = ca3_buffer[buf_idx];
            buf_idx = (buf_idx + 1) % BUFFER_SIZE;
        } else {
            buf_idx = (seed_idx + 1) % BUFFER_SIZE;
        }
    }

    replay_active = true;
}

std::vector<float> ReplayEngine::get_replay_frame() {
    if (!replay_active || replay_pos >= static_cast<int>(replay_sequence.size())) {
        replay_active = false;
        return {};
    }
    return replay_sequence[replay_pos++];
}

void ReplayEngine::finalize_replay() {
    replay_active = false;
    replay_pos = 0;
    replay_sequence.clear();
}