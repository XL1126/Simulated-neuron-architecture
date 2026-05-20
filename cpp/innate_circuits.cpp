#include "innate_circuits.h"
#include <algorithm>

InnateCircuitBuilder::InnateCircuitBuilder(uint32_t seed) {
    rng.seed(seed);
}

float InnateCircuitBuilder::_gabor(float x, float y, float theta,
                                    float sigma, float freq, float phase) {
    float xr = x * std::cos(theta) + y * std::sin(theta);
    float yr = -x * std::sin(theta) + y * std::cos(theta);
    float gaussian = std::exp(-(xr*xr + yr*yr) / (2.0f * sigma * sigma));
    return gaussian * std::cos(2.0f * 3.14159f * freq * xr + phase);
}

float InnateCircuitBuilder::_dog(float x, float y, float sigma_c, float sigma_s) {
    float center = std::exp(-(x*x + y*y) / (2.0f * sigma_c * sigma_c));
    float surround = std::exp(-(x*x + y*y) / (2.0f * sigma_s * sigma_s));
    return center - 0.7f * surround;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_visual_retinotopic(uint32_t base_id, uint32_t n,
                                                uint32_t field_w, uint32_t field_h) {
    std::vector<InbornSynapse> synapses;
    for (uint32_t i = 0; i < n; i++) {
        InbornSynapse s;
        s.src = base_id + i;
        s.dst = base_id + ((i + field_w/2) % n);
        s.weight = 0.3f + _dog((float)(i % field_w) - field_w/2.0f,
                                (float)(i / field_w) - field_h/2.0f,
                                1.5f, 3.0f) * 0.2f;
        s.weight = std::max(0.01f, std::min(0.8f, s.weight));
        s.delay = 1 + (uint8_t)(i % 3);
        s.is_core = true;
        synapses.push_back(s);
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_visual_orientation_columns(uint32_t base_id, uint32_t n,
                                                        uint32_t n_orientations) {
    std::vector<InbornSynapse> synapses;
    for (uint32_t i = 0; i < n; i++) {
        float theta = 3.14159f * (float)(i % n_orientations) / (float)n_orientations;
        for (uint32_t j = 0; j < n; j++) {
            if (i == j) continue;
            float sim = _gabor((float)(i % 16), (float)(j % 16), theta, 2.0f, 0.3f, 0.0f);
            if (std::abs(sim) > 0.1f) {
                InbornSynapse s;
                s.src = base_id + i;
                s.dst = base_id + j;
                s.weight = std::max(0.01f, std::min(0.5f, std::abs(sim) * 2.0f));
                s.delay = (uint8_t)(1 + (uint32_t)(std::abs(sim) * 4));
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_motor_somatotopic(uint32_t base_id, uint32_t n,
                                               uint32_t n_actions, uint32_t n_directions) {
    std::vector<InbornSynapse> synapses;
    uint32_t neurons_per_action = n / n_actions;
    if (neurons_per_action == 0) neurons_per_action = 1;

    for (uint32_t act = 0; act < n_actions; act++) {
        uint32_t act_start = base_id + act * neurons_per_action;
        uint32_t act_end = std::min(act_start + neurons_per_action, base_id + n);

        for (uint32_t i = act_start; i < act_end; i++) {
            for (uint32_t j = i + 1; j < act_end; j++) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.15f;
                s.delay = 1;
                s.is_core = true;
                synapses.push_back(s);

                s.src = j; s.dst = i;
                synapses.push_back(s);
            }
        }

        for (uint32_t i = act_start; i < act_end; i++) {
            uint32_t other_start = base_id + ((act + 1) % n_actions) * neurons_per_action;
            uint32_t other_end = std::min(other_start + neurons_per_action/2, base_id + n);
            for (uint32_t j = other_start; j < other_end; j++) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.03f;
                s.delay = 2 + (uint8_t)(act % 3);
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_hippocampal_dg_ca3_ca1(uint32_t dg_base, uint32_t dg_n,
                                                     uint32_t ca3_base, uint32_t ca3_n,
                                                     uint32_t ca1_base, uint32_t ca1_n) {
    std::vector<InbornSynapse> synapses;

    for (uint32_t i = dg_base; i < dg_base + dg_n; i++) {
        uint32_t ca3_target = ca3_base + ((i - dg_base) * 7) % ca3_n;
        InbornSynapse s;
        s.src = i; s.dst = ca3_target;
        s.weight = 0.2f;
        s.delay = 1;
        s.is_core = true;
        synapses.push_back(s);
    }

    for (uint32_t i = ca3_base; i < ca3_base + ca3_n; i++) {
        for (uint32_t j_offset = 0; j_offset < 12; j_offset++) {
            uint32_t j = ca3_base + ((i - ca3_base) + j_offset * 73 + (i % 31)) % ca3_n;
            if (i != j) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.08f;
                s.delay = 1 + (uint8_t)(j_offset % 3);
                s.is_core = false;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t i = ca3_base; i < ca3_base + ca3_n; i++) {
        uint32_t ca1_target = ca1_base + ((i - ca3_base) * 3) % ca1_n;
        InbornSynapse s;
        s.src = i; s.dst = ca1_target;
        s.weight = 0.15f;
        s.delay = 1;
        s.is_core = true;
        synapses.push_back(s);
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_prefrontal_working_memory(uint32_t base_id, uint32_t n,
                                                       uint32_t n_slots) {
    std::vector<InbornSynapse> synapses;
    uint32_t per_slot = n / n_slots;
    if (per_slot == 0) per_slot = 1;

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        uint32_t start = base_id + slot * per_slot;
        uint32_t end = std::min(start + per_slot, base_id + n);

        for (uint32_t i = start; i < end; i++) {
            for (uint32_t j = start; j < end; j++) {
                if (i == j) continue;
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.12f + 0.05f * (float)((i + j) % 3);
                s.delay = (uint8_t)(1 + (i % 3));
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t i = base_id; i < base_id + n; i++) {
        uint32_t i_slot = ((i - base_id) / per_slot) % n_slots;
        for (uint32_t j = base_id; j < base_id + n; j++) {
            uint32_t j_slot = ((j - base_id) / per_slot) % n_slots;
            if (i_slot != j_slot) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = -0.08f;
                s.delay = 2;
                s.is_core = false;
                synapses.push_back(s);
            }
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_amygdala_valence(uint32_t base_id, uint32_t n) {
    std::vector<InbornSynapse> synapses;
    uint32_t mid = n / 2;

    for (uint32_t i = base_id; i < base_id + mid; i++) {
        for (uint32_t j = base_id; j < i; j++) {
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.2f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
            s.src = j; s.dst = i;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = base_id + mid; i < base_id + n; i++) {
        for (uint32_t j = base_id + mid; j < i; j++) {
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.2f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
            s.src = j; s.dst = i;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = base_id; i < base_id + mid; i++) {
        for (uint32_t j = base_id + mid; j < base_id + n; j++) {
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = -0.15f;
            s.delay = 2;
            s.is_core = true;
            synapses.push_back(s);
            s.src = j; s.dst = i;
            s.weight = -0.15f;
            synapses.push_back(s);
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_language_semantic(uint32_t base_id, uint32_t n,
                                               const std::vector<std::string>& concepts) {
    std::vector<InbornSynapse> synapses;
    if (concepts.empty()) return synapses;
    uint32_t per_concept = n / (uint32_t)concepts.size();
    if (per_concept < 4) per_concept = 4;

    std::hash<std::string> hasher;

    for (size_t ci = 0; ci < concepts.size(); ci++) {
        uint32_t start = base_id + (uint32_t)ci * per_concept;
        uint32_t end = std::min(start + per_concept, base_id + n);

        for (uint32_t i = start; i < end; i++) {
            for (uint32_t j = start; j < end; j++) {
                if (i == j) continue;
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.1f;
                s.delay = 1;
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    for (size_t ci = 0; ci < concepts.size(); ci++) {
        for (size_t cj = ci + 1; cj < concepts.size(); cj++) {
            uint32_t h1 = (uint32_t)(hasher(concepts[ci]) % 1000);
            uint32_t h2 = (uint32_t)(hasher(concepts[cj]) % 1000);
            float similarity = 1.0f / (1.0f + (float)std::abs((int)(h1 - h2)) / 100.0f);

            if (similarity > 0.05f) {
                uint32_t si = base_id + (uint32_t)ci * per_concept;
                uint32_t sj = base_id + (uint32_t)cj * per_concept;
                for (uint32_t k = 0; k < std::min(per_concept/2, (uint32_t)3); k++) {
                    InbornSynapse s;
                    s.src = si + k;
                    s.dst = sj + k;
                    s.weight = std::max(0.005f, similarity * 0.05f);
                    s.delay = (uint8_t)(1 + (uint32_t)(similarity * 5));
                    s.is_core = false;
                    synapses.push_back(s);
                }
            }
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_global_workspace(uint32_t base_id, uint32_t n) {
    std::vector<InbornSynapse> synapses;

    for (uint32_t i = base_id; i < base_id + n; i++) {
        for (uint32_t j = base_id; j < base_id + n; j++) {
            if (i == j) continue;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = -0.3f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = base_id; i < base_id + n; i++) {
        InbornSynapse self;
        self.src = i; self.dst = i;
        self.weight = 0.8f;
        self.delay = 1;
        self.is_core = true;
        synapses.push_back(self);
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_inter_region(uint32_t src_base, uint32_t src_n,
                                          uint32_t dst_base, uint32_t dst_n,
                                          float prob, float max_weight, uint8_t max_delay) {
    std::vector<InbornSynapse> synapses;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(0.01f, max_weight);

    for (uint32_t i = 0; i < src_n; i++) {
        if (dist(rng) < prob) {
            InbornSynapse s;
            s.src = src_base + i;
            s.dst = dst_base + (i * 3 + (uint32_t)(dist(rng) * 1000)) % dst_n;
            s.weight = wdist(rng);
            s.delay = (uint8_t)(1 + (uint32_t)(dist(rng) * max_delay));
            s.is_core = false;
            synapses.push_back(s);
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_thalamic_relay(uint32_t base_id, uint32_t n) {
    std::vector<InbornSynapse> synapses;
    uint32_t n_groups = std::max((uint32_t)1, n / 16);
    uint32_t per_group = n / n_groups;

    for (uint32_t g = 0; g < n_groups; g++) {
        uint32_t start = base_id + g * per_group;
        uint32_t end = std::min(start + per_group, base_id + n);
        for (uint32_t i = start; i < end; i++) {
            for (uint32_t j = start; j < end; j++) {
                if (i == j) continue;
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.08f;
                s.delay = 1;
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t g = 0; g < n_groups - 1; g++) {
        uint32_t s1 = base_id + g * per_group;
        uint32_t s2 = base_id + (g + 1) * per_group;
        for (uint32_t k = 0; k < std::min(per_group / 2, (uint32_t)4); k++) {
            InbornSynapse s;
            s.src = s1 + k; s.dst = s2 + k;
            s.weight = 0.05f; s.delay = 2; s.is_core = true;
            synapses.push_back(s);
            s.src = s2 + k; s.dst = s1 + k;
            synapses.push_back(s);
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_claustrum_coordinator(uint32_t base_id, uint32_t n) {
    std::vector<InbornSynapse> synapses;
    for (uint32_t i = base_id; i < base_id + n; i++) {
        for (uint32_t j = base_id; j < base_id + n; j++) {
            if (i == j) continue;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = (std::abs((int)(i - j)) < (int)(n / 4)) ? 0.12f : -0.05f;
            s.delay = (uint8_t)(1 + ((i + j) % 3));
            s.is_core = true;
            synapses.push_back(s);
        }
    }
    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_default_mode_network(uint32_t base_id, uint32_t n) {
    std::vector<InbornSynapse> synapses;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = base_id; i < base_id + n; i++) {
        uint32_t n_conn = 8 + (uint32_t)(dist(rng) * 12);
        for (uint32_t k = 0; k < n_conn; k++) {
            uint32_t j = base_id + ((i - base_id) + (k * 67 + 13) % (n / 2) + n / 4) % n;
            if (i != j) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.04f + dist(rng) * 0.08f;
                s.delay = (uint8_t)(1 + (k % 5));
                s.is_core = false;
                synapses.push_back(s);
            }
        }
    }

    uint32_t hub_center = base_id + n / 2;
    for (uint32_t i = base_id; i < base_id + n; i++) {
        if (i == hub_center) continue;
        InbornSynapse s;
        s.src = hub_center; s.dst = i;
        s.weight = 0.15f; s.delay = 1; s.is_core = true;
        synapses.push_back(s);
        s.src = i; s.dst = hub_center;
        s.weight = 0.08f;
        synapses.push_back(s);
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_workspace_three_layer(uint32_t base_id, uint32_t n_total,
                                                   uint32_t n_sensory, uint32_t n_competition,
                                                   uint32_t n_broadcast) {
    std::vector<InbornSynapse> synapses;
    uint32_t sensory_base = base_id;
    uint32_t competition_base = base_id + n_sensory;
    uint32_t broadcast_base = base_id + n_sensory + n_competition;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = sensory_base; i < sensory_base + n_sensory; i++) {
        for (uint32_t k = 0; k < 5; k++) {
            uint32_t j = competition_base + ((i - sensory_base) * 3 + k * 71) % n_competition;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.15f + dist(rng) * 0.1f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    uint32_t k_wta = (uint32_t)std::sqrt((float)n_competition);
    for (uint32_t i = competition_base; i < competition_base + n_competition; i++) {
        for (uint32_t j = competition_base; j < competition_base + n_competition; j++) {
            if (i == j) {
                InbornSynapse self;
                self.src = i; self.dst = i;
                self.weight = 0.6f; self.delay = 1; self.is_core = true;
                synapses.push_back(self);
            } else {
                float dist_idx = (float)std::abs((int)(i - j));
                float inhib = -0.15f / (1.0f + dist_idx / (float)k_wta);
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = inhib;
                s.delay = 1;
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t i = competition_base; i < competition_base + n_competition; i++) {
        for (uint32_t k = 0; k < 8; k++) {
            uint32_t j = broadcast_base + ((i - competition_base) + k * 43) % n_broadcast;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.2f + dist(rng) * 0.1f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = broadcast_base; i < broadcast_base + n_broadcast; i++) {
        for (uint32_t k = 0; k < 6; k++) {
            uint32_t j = broadcast_base + ((i - broadcast_base) + k * 31 + 7) % n_broadcast;
            if (i != j) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.1f + dist(rng) * 0.05f;
                s.delay = (uint8_t)(1 + (k % 3));
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_hippocampal_structured(uint32_t dg_base, uint32_t dg_n,
                                                     uint32_t ca3_base, uint32_t ca3_n,
                                                     uint32_t ca1_base, uint32_t ca1_n,
                                                     float dg_sparsity, float ca3_recurrent_prob) {
    std::vector<InbornSynapse> synapses;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = dg_base; i < dg_base + dg_n; i++) {
        for (uint32_t j = dg_base; j < dg_base + dg_n; j++) {
            if (i == j) continue;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = -0.12f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    uint32_t dg_active = std::max((uint32_t)1, (uint32_t)(dg_n * dg_sparsity));
    for (uint32_t i = dg_base; i < dg_base + dg_n; i++) {
        uint32_t n_proj = std::max((uint32_t)2, ca3_n / dg_active / 2);
        for (uint32_t k = 0; k < n_proj; k++) {
            uint32_t j = ca3_base + ((i - dg_base) * (k + 1) * 7 + k * 31) % ca3_n;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.25f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = ca3_base; i < ca3_base + ca3_n; i++) {
        uint32_t n_recurrent = (uint32_t)(ca3_n * ca3_recurrent_prob);
        for (uint32_t k = 0; k < n_recurrent; k++) {
            uint32_t j = ca3_base + ((i - ca3_base) + k * 73 + (i % 31)) % ca3_n;
            if (i != j) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = 0.06f;
                s.delay = (uint8_t)(1 + (k % 3));
                s.is_core = false;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t i = ca3_base; i < ca3_base + ca3_n; i++) {
        uint32_t j = ca1_base + ((i - ca3_base) * 3) % ca1_n;
        InbornSynapse s;
        s.src = i; s.dst = j;
        s.weight = 0.18f;
        s.delay = 1;
        s.is_core = true;
        synapses.push_back(s);
    }

    for (uint32_t i = ca1_base; i < ca1_base + ca1_n; i++) {
        uint32_t j = ca1_base + ((i - ca1_base) + ca1_n / 3) % ca1_n;
        InbornSynapse s;
        s.src = i; s.dst = j;
        s.weight = 0.1f;
        s.delay = 2;
        s.is_core = true;
        synapses.push_back(s);
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_visual_hierarchical(uint32_t v1_base, uint32_t v1_n,
                                                 uint32_t v2_base, uint32_t v2_n,
                                                 uint32_t n_orientations) {
    std::vector<InbornSynapse> synapses;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = v1_base; i < v1_base + v1_n; i++) {
        InbornSynapse s;
        s.src = i;
        s.dst = v1_base + ((i - v1_base + v1_n / 4) % v1_n);
        s.weight = 0.3f + _dog((float)((i - v1_base) % 16) - 8.0f,
                                (float)((i - v1_base) / 16) - 8.0f,
                                1.5f, 3.0f) * 0.2f;
        s.weight = std::max(0.01f, std::min(0.8f, s.weight));
        s.delay = 1 + (uint8_t)((i - v1_base) % 3);
        s.is_core = true;
        synapses.push_back(s);
    }

    for (uint32_t i = v1_base; i < v1_base + v1_n; i++) {
        float theta = 3.14159f * (float)((i - v1_base) % n_orientations) / (float)n_orientations;
        for (uint32_t j = v1_base; j < v1_base + v1_n; j++) {
            if (i == j) continue;
            float sim = _gabor((float)((i - v1_base) % 16), (float)((j - v1_base) % 16),
                              theta, 2.0f, 0.3f, 0.0f);
            if (std::abs(sim) > 0.1f) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = std::max(0.01f, std::min(0.5f, std::abs(sim) * 2.0f));
                s.delay = (uint8_t)(1 + (uint32_t)(std::abs(sim) * 4));
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    for (uint32_t i = v1_base; i < v1_base + v1_n; i++) {
        uint32_t n_v2_targets = 3;
        for (uint32_t k = 0; k < n_v2_targets; k++) {
            uint32_t j = v2_base + ((i - v1_base) * (k + 1) + k * 41) % v2_n;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = 0.12f + dist(rng) * 0.08f;
            s.delay = (uint8_t)(1 + (k % 2));
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = v2_base; i < v2_base + v2_n; i++) {
        uint32_t pool_size = std::max((uint32_t)1, v2_n / 16);
        uint32_t pool_start = v2_base + ((i - v2_base) / pool_size) * pool_size;
        uint32_t pool_end = std::min(pool_start + pool_size, v2_base + v2_n);
        for (uint32_t j = pool_start; j < pool_end; j++) {
            if (i == j) continue;
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = -0.1f;
            s.delay = 1;
            s.is_core = true;
            synapses.push_back(s);
        }
        for (uint32_t j = pool_start; j < pool_end; j++) {
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = (i == j) ? 0.4f : 0.06f;
            s.delay = (uint8_t)(1 + ((i + j) % 3));
            s.is_core = true;
            synapses.push_back(s);
        }
    }

    return synapses;
}

std::vector<InbornSynapse>
InnateCircuitBuilder::build_prefrontal_esn(uint32_t base_id, uint32_t n_neurons,
                                            uint32_t reservoir_size, uint32_t readout_size) {
    std::vector<InbornSynapse> synapses;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> ndist(0.0f, 0.3f);

    uint32_t reservoir_start = base_id;
    uint32_t reservoir_end = base_id + std::min(reservoir_size, n_neurons);
    uint32_t readout_start = reservoir_end;
    uint32_t readout_end = std::min(readout_start + readout_size, base_id + n_neurons);

    float spectral_radius = 0.9f;
    for (uint32_t i = reservoir_start; i < reservoir_end; i++) {
        for (uint32_t k = 0; k < 20; k++) {
            uint32_t j = reservoir_start + ((i - reservoir_start) + k * 47 + 11) % (reservoir_end - reservoir_start);
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = ndist(rng) * spectral_radius / 20.0f;
            s.delay = (uint8_t)(1 + (k % 3));
            s.is_core = false;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = reservoir_start; i < reservoir_end; i++) {
        for (uint32_t j = readout_start; j < readout_end; j++) {
            InbornSynapse s;
            s.src = i; s.dst = j;
            s.weight = ndist(rng) * 0.1f;
            s.delay = 1;
            s.is_core = false;
            synapses.push_back(s);
        }
    }

    for (uint32_t i = readout_start; i < readout_end; i++) {
        for (uint32_t k = 0; k < 8; k++) {
            uint32_t j = readout_start + ((i - readout_start) + k * 13 + 3) % (readout_end - readout_start);
            if (i != j) {
                InbornSynapse s;
                s.src = i; s.dst = j;
                s.weight = -0.08f;
                s.delay = 1;
                s.is_core = true;
                synapses.push_back(s);
            }
        }
    }

    return synapses;
}