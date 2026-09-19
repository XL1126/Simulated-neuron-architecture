#pragma once
#include <vector>
#include <cstdint>
#include <cmath>

struct ReplayEngine {
    static const int BUFFER_SIZE = 200;
    std::vector<std::vector<float>> ca3_buffer;
    int buffer_write_pos;
    int buffer_count;

    std::vector<std::vector<float>> replay_sequence;
    int replay_pos;
    bool replay_active;

    ReplayEngine();
    void init(size_t ca3_size);

    void record_ca3_snapshot(const std::vector<float>& ca3_activity);
    void init_replay_sequence(size_t ca3_size);
    std::vector<float> get_replay_frame();
    bool is_active() const { return replay_active; }
    int total_frames() const { return static_cast<int>(replay_sequence.size()); }
    int current_pos() const { return replay_pos; }
    void finalize_replay();
};