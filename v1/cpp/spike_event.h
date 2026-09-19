#pragma once
#include <cstdint>

struct SpikeEvent {
    uint32_t src_id;
    uint32_t dst_id;
    float strength;
    uint16_t delay_ms;
    uint64_t time_step;
};

struct SpikeRecord {
    uint64_t time_step;
    float strength;
};

struct FireHistoryEntry {
    uint64_t time_step;
    uint32_t src_id;
    uint32_t dst_id;
    float strength;
};
