#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

class SelfPerceptionNetwork {
public:
    static constexpr size_t SELF_DIM = 128;
    static constexpr size_t HIDDEN_DIM = 256;

    static constexpr size_t MOTOR_DIM = 32;
    static constexpr size_t AMYGDALA_DIM = 16;
    static constexpr size_t MEMORY_DIM = 32;
    static constexpr size_t INTENTION_DIM = 32;
    static constexpr size_t TOTAL_INPUT_DIM = MOTOR_DIM + AMYGDALA_DIM + MEMORY_DIM + INTENTION_DIM;

    SelfPerceptionNetwork();

    void init();

    std::vector<float> forward(const std::vector<float>& motor_efference,
                                const std::vector<float>& amygdala_emotion,
                                const std::vector<float>& memory_recall,
                                const std::vector<float>& prefrontal_intentions);

    void train_step(const std::vector<float>& target, float lr);

    const std::vector<float>& get_self_state() const { return self_state; }

private:
    std::vector<float> W1;
    std::vector<float> b1;
    std::vector<float> W2;
    std::vector<float> b2;

    std::vector<float> ln1_gamma;
    std::vector<float> ln1_beta;

    std::vector<float> h1;
    std::vector<float> h1_norm;
    std::vector<float> h1_act;
    std::vector<float> self_state;

    void layer_norm(const std::vector<float>& x, std::vector<float>& out,
                     const std::vector<float>& gamma, const std::vector<float>& beta);
    void fc_forward(const std::vector<float>& x, std::vector<float>& out,
                     const std::vector<float>& W, const std::vector<float>& b,
                     size_t in_dim, size_t out_dim);
    void relu(std::vector<float>& x);
};