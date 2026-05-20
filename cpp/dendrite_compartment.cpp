#include "dendrite_compartment.h"
#include <algorithm>
#include <cmath>

float DendriteCompartment::compute_output() const {
    if (input_window.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : input_window) sum += v;
    float avg = sum / static_cast<float>(input_window.size());
    return coeff * (2.0f / (1.0f + std::exp(-sigmoid_steepness * avg / 0.5f)) - 1.0f);
}

void DendriteCompartment::add_input(float strength) {
    input_window.push_back(std::max(-1.0f, std::min(1.0f, strength)));
}

void DendriteCompartment::decay_window() {
    while (input_window.size() > window_size) input_window.pop_front();
}

void DendriteTree::init(uint32_t num_compartments, uint32_t window_sz) {
    compartments.resize(num_compartments);
    for (uint32_t i = 0; i < num_compartments; i++) {
        compartments[i].window_size = window_sz;
        compartments[i].coeff = 1.0f / std::sqrt(static_cast<float>(num_compartments));
        compartments[i].distance_from_soma = 50.0f + static_cast<float>(i) * 25.0f;
    }
}

float DendriteTree::compute_total_output() {
    float total = 0.0f;
    for (auto& comp : compartments) {
        total += comp.compute_output();
    }
    return total;
}

void DendriteTree::add_input_distributed(float strength) {
    for (auto& comp : compartments) {
        comp.add_input(strength * (0.8f + 0.2f * static_cast<float>(rand()) / RAND_MAX));
    }
}

void DendriteTree::step() {
    for (auto& comp : compartments) {
        comp.decay_window();
        comp.add_input(0.0f);
        if (comp.input_window.size() > comp.window_size) comp.decay_window();
    }
}

void DendriteTree::update_calcium_on_all(float vm, float dt_ms) {
    for (auto& comp : compartments) {
        comp.update_calcium(vm, dt_ms);
    }
}