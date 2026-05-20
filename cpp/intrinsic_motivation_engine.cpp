#include "intrinsic_motivation_engine.h"

static inline bool is_bad(float x) { return !std::isfinite(x); }

IntrinsicMotivationEngine::IntrinsicMotivationEngine()
    : novelty_drive(0.3f), competence_drive(0.15f), self_consistency_drive(0.2f),
      composite_intrinsic_reward(0.0f), td_error(0.0f), td_error_ema(0.0f),
      prev_value_estimate(0.0f), steps(0),
      exploration_temperature(1.0f), exploitation_ratio(0.3f)
{
    region_pred_error.assign(N_REGIONS, 0.1f);
    region_novelty.assign(N_REGIONS, 0.0f);
}

void IntrinsicMotivationEngine::init(size_t self_dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    self_reference.assign(self_dim, 0.0f);

    for (int r = 0; r < N_REGIONS; r++) {
        region_ema[r].assign(PREDICTOR_DIM, 0.0f);
        region_predictor[r].assign(PREDICTOR_DIM, dist(rng));
    }

    global_error_history.clear();
    reward_history.clear();
    region_pred_error.assign(N_REGIONS, 0.1f);
    region_novelty.assign(N_REGIONS, 0.0f);
    novelty_drive = 0.3f;
    competence_drive = 0.15f;
    self_consistency_drive = 0.2f;
    composite_intrinsic_reward = 0.0f;
    steps = 0;
}

void IntrinsicMotivationEngine::update_region_prediction(
    int region_idx,
    const std::vector<float>& region_activity,
    size_t n_dims)
{
    if (region_idx < 0 || region_idx >= N_REGIONS) return;
    if (region_activity.empty()) return;

    std::vector<float> compressed(PREDICTOR_DIM, 0.0f);
    size_t seg = n_dims / PREDICTOR_DIM;
    if (seg == 0) seg = 1;
    for (size_t i = 0; i < PREDICTOR_DIM && i * seg < region_activity.size(); i++) {
        float acc = 0.0f;
        size_t count = 0;
        for (size_t j = i * seg; j < (i + 1) * seg && j < region_activity.size(); j++) {
            if (!is_bad(region_activity[j])) {
                acc += region_activity[j];
                count++;
            }
        }
        compressed[i] = count > 0 ? acc / (float)count : 0.0f;
    }

    float pred_error = 0.0f;
    float norm = 0.0f;
    for (size_t i = 0; i < PREDICTOR_DIM; i++) {
        if (region_ema[region_idx].size() <= i
            || region_predictor[region_idx].size() <= i) break;
        float pred = region_ema[region_idx][i] * region_predictor[region_idx][i];
        float actual = compressed[i];
        if (!is_bad(actual)) {
            float err = actual - pred;
            pred_error += err * err;
            norm += 1.0f;
        }
    }
    if (norm > 0.0f) {
        pred_error = std::sqrt(pred_error / norm);
        region_pred_error[region_idx] = region_pred_error[region_idx] * 0.8f + pred_error * 0.2f;
    }

    float alpha = 0.05f;
    for (size_t i = 0; i < PREDICTOR_DIM; i++) {
        float act = compressed[i];
        if (!is_bad(act)) {
            if (region_ema[region_idx].size() > i)
                region_ema[region_idx][i] = region_ema[region_idx][i] * (1.0f - alpha) + act * alpha;
        }
    }

    float mean_err = 0.0f;
    for (int r = 0; r < N_REGIONS; r++) mean_err += region_pred_error[r];
    mean_err /= (float)N_REGIONS;
    region_novelty[region_idx] = std::abs(region_pred_error[region_idx] - mean_err)
        / std::max(0.001f, mean_err);
}

float IntrinsicMotivationEngine::compute(
    float world_pred_error, float self_model_error,
    const std::vector<float>& self_state,
    const std::vector<float>& ama_self_prototype,
    float external_reward)
{
    steps++;

    global_error_history.push_back(world_pred_error);
    if ((int)global_error_history.size() > HISTORY_LEN) global_error_history.pop_front();

    float mean_err = 0.0f;
    for (auto e : global_error_history) mean_err += e;
    mean_err /= std::max(1.0f, (float)global_error_history.size());

    float novelty = 0.0f;
    if (global_error_history.size() >= 10) {
        float recent_err = 0.0f;
        int cnt = 0;
        auto it = global_error_history.rbegin();
        for (int i = 0; i < 5 && it != global_error_history.rend(); i++, ++it) {
            recent_err += *it;
            cnt++;
        }
        if (cnt > 0) {
            recent_err /= (float)cnt;
            float older_err = 0.0f;
            cnt = 0;
            for (int i = 0; i < 5 && it != global_error_history.rend(); i++, ++it) {
                older_err += *it;
                cnt++;
            }
            if (cnt > 0) {
                older_err /= (float)cnt;
                float diff = older_err - recent_err;
                novelty = diff / std::max(0.001f, older_err + recent_err);
                if (novelty < -1.0f) novelty = -1.0f;
                if (novelty > 1.0f) novelty = 1.0f;
                novelty = (novelty + 1.0f) * 0.5f;
            }
        }
    }

    float region_novelty_avg = 0.0f;
    for (int r = 0; r < N_REGIONS; r++) region_novelty_avg += region_novelty[r];
    region_novelty_avg /= (float)N_REGIONS;

    float competence = 1.0f - std::min(1.0f, self_model_error * 2.0f);

    float ama_consistency = 0.5f;
    if (!ama_self_prototype.empty() && !self_state.empty()) {
        float dist = 0.0f;
        size_t n = std::min(ama_self_prototype.size(), self_state.size());
        for (size_t i = 0; i < n; i++) {
            float d = ama_self_prototype[i] - self_state[i];
            dist += d * d;
        }
        dist = std::sqrt(dist / (float)n);
        ama_consistency = 1.0f / (1.0f + dist * 2.0f);
    }

    float self_ref_dist = 0.0f;
    if (self_reference.size() == self_state.size() && steps > 20) {
        for (size_t i = 0; i < self_state.size(); i++) {
            float d = self_state[i] - self_reference[i];
            self_ref_dist += d * d;
        }
        self_ref_dist = std::sqrt(self_ref_dist / (float)self_state.size());
        float alpha = 0.005f;
        for (size_t i = 0; i < self_state.size(); i++) {
            self_reference[i] = self_reference[i] * (1.0f - alpha) + self_state[i] * alpha;
        }
    } else if (self_reference.empty() || self_reference.size() != self_state.size()) {
        self_reference = self_state;
    }
    float self_consistency = 1.0f / (1.0f + self_ref_dist * 3.0f);

    novelty_drive = novelty_drive * 0.95f + (novelty * 0.3f + region_novelty_avg * 0.3f) * 0.05f;
    competence_drive = competence_drive * 0.95f + competence * 0.05f;
    self_consistency_drive = self_consistency_drive * 0.95f
        + (ama_consistency * 0.5f + self_consistency * 0.5f) * 0.05f;

    float value_estimate = novelty_drive * 0.35f
        + competence_drive * 0.25f
        + self_consistency_drive * 0.25f
        + external_reward * 0.15f;

    td_error = value_estimate - prev_value_estimate + external_reward * 0.3f;
    td_error_ema = td_error_ema * 0.9f + td_error * 0.1f;
    prev_value_estimate = value_estimate;

    composite_intrinsic_reward = novelty_drive * 0.4f
        + competence_drive * 0.3f
        + self_consistency_drive * 0.3f;

    if (composite_intrinsic_reward > 1.0f) composite_intrinsic_reward = 1.0f;
    if (composite_intrinsic_reward < 0.0f) composite_intrinsic_reward = 0.0f;

    exploration_temperature = 0.5f + novelty_drive * 1.5f;
    exploitation_ratio = 0.1f + competence_drive * 0.6f;

    return composite_intrinsic_reward;
}

void IntrinsicMotivationEngine::generate_curiosity_options(
    const std::vector<std::vector<float>>& region_states,
    const std::vector<int>& memory_indices)
{
    for (int i = 0; i < N_OPTIONS; i++) {
        int kind = i % 3;
        CuriosityOption& opt = candidate_options[i];

        if (kind == 0) {
            opt.target_kind = 0;
            opt.target_index = i % std::max(1, (int)region_states.size());
            if (!region_states.empty()
                && opt.target_index < (int)region_states.size()
                && opt.target_index < N_REGIONS) {
                opt.novelty_score = compute_novelty_bonus(
                    region_states[opt.target_index], region_ema[opt.target_index]);
                opt.info_gain = compute_info_gain(
                    region_states[opt.target_index], region_predictor[opt.target_index]);
            } else {
                opt.novelty_score = 0.3f;
                opt.info_gain = 0.2f;
            }
        } else if (kind == 1) {
            opt.target_kind = 1;
            opt.target_index = i % std::max(1, (int)memory_indices.size());
            opt.novelty_score = 0.3f + (float)(rand() % 100) * 0.004f;
            opt.info_gain = 0.2f + (float)(rand() % 100) * 0.003f;
        } else {
            opt.target_kind = 2;
            opt.target_index = i;
            opt.novelty_score = 0.1f + (float)(rand() % 100) * 0.002f;
            opt.info_gain = 0.1f + (float)(rand() % 100) * 0.002f;
        }

        opt.competence_gain = 0.1f;
        opt.total_score = opt.novelty_score * novelty_drive
            + opt.info_gain * 0.3f
            + opt.competence_gain * competence_drive;
    }
}

int IntrinsicMotivationEngine::select_curiosity_action(float temperature) {
    if (temperature <= 0.0f) temperature = 0.1f;

    float scores[N_OPTIONS];
    float max_score = -1e9f;
    for (int i = 0; i < N_OPTIONS; i++) {
        scores[i] = candidate_options[i].total_score;
        if (scores[i] > max_score) max_score = scores[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < N_OPTIONS; i++) {
        scores[i] = std::exp((scores[i] - max_score) / temperature);
        sum += scores[i];
    }

    if (sum <= 0.0f) return 0;

    float r = (float)(rand() % 10000) / 10000.0f;
    float cum = 0.0f;
    for (int i = 0; i < N_OPTIONS; i++) {
        cum += scores[i] / sum;
        if (r <= cum) return i;
    }
    return N_OPTIONS - 1;
}

float IntrinsicMotivationEngine::compute_novelty_bonus(
    const std::vector<float>& state, const std::vector<float>& history_ema)
{
    if (state.empty() || history_ema.empty()) return 0.5f;
    float kl = 0.0f;
    size_t n = std::min(state.size(), history_ema.size());
    for (size_t i = 0; i < n; i++) {
        float p = std::max(0.001f, std::abs(state[i]));
        float q = std::max(0.001f, std::abs(history_ema[i]));
        kl += p * std::log(p / q);
    }
    kl /= (float)n;
    return std::min(1.0f, kl * 0.5f);
}

float IntrinsicMotivationEngine::compute_info_gain(
    const std::vector<float>& state, const std::vector<float>& predictor)
{
    if (state.empty() || predictor.empty()) return 0.3f;
    float err = 0.0f;
    size_t n = std::min(state.size(), predictor.size());
    for (size_t i = 0; i < n; i++) {
        float d = state[i] - predictor[i];
        err += d * d;
    }
    err = std::sqrt(err / (float)n);
    return std::min(1.0f, err * 2.0f);
}