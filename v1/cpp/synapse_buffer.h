#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct SynapseSOA {
    std::vector<float> weight;
    std::vector<int32_t> pre_neuron;
    std::vector<int32_t> post_neuron;
    std::vector<float> tag;
    std::vector<float> eligibility_trace;
    std::vector<float> d1_density;
    std::vector<float> d2_density;
    std::vector<uint8_t> delay;
    std::vector<uint8_t> is_core;
    std::vector<float> ltp_e_fast;
    std::vector<float> ltp_e_medium;
    std::vector<float> ltp_e_slow;
    std::vector<float> ltd_e_fast;
    std::vector<float> ltd_e_medium;
    std::vector<float> ltd_e_slow;

    size_t size() const { return weight.size(); }
    void reserve(size_t n);

    void clear();
};

struct NeuronSOA {
    std::vector<float> v;
    std::vector<float> u;
    std::vector<float> I_syn;
    std::vector<float> I_dendrite;
    std::vector<float> I_bias;
    std::vector<float> I_noise;
    std::vector<float> threshold;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> avg_firing_rate;
    std::vector<uint8_t> fired;
    std::vector<float> r1, r2, o1, o2;
    std::vector<float> ca_concentration;
    std::vector<int32_t> out_begin;
    std::vector<int32_t> out_count;
    std::vector<int32_t> fire_count;

    void resize(size_t n);
    size_t size() const { return v.size(); }
};