#include "goal_generator.h"
#include <sstream>

GoalGenerator::GoalGenerator()
    : n_active_goals(0), current_goal_idx(-1),
      goal_history_embedding(EMBED_DIM, 0.0f),
      achievement_memory(EMBED_DIM, 0.0f),
      goal_satisfaction(0.0f), goal_drive(0.3f),
      exploration_curiosity(0.5f), persistence(0.5f),
      consecutive_failures(0), step_counter(0)
{
    for (int i = 0; i < MAX_GOALS; i++) {
        active_goals[i].goal_state.assign(GOAL_DIM, 0.0f);
        active_goals[i].goal_embedding.assign(EMBED_DIM, 0.0f);
        active_goals[i].value = 0.0f;
        active_goals[i].progress = 0.0f;
        active_goals[i].difficulty_estimate = 0.5f;
        active_goals[i].novelty = 0.5f;
        active_goals[i].competence_affordance = 0.3f;
        active_goals[i].self_consistency = 0.5f;
        active_goals[i].total_desirability = 0.0f;
        active_goals[i].timescale = MEDIUM_TERM;
        active_goals[i].created_at = 0;
        active_goals[i].last_pursued = 0;
        active_goals[i].pursue_count = 0;
        active_goals[i].is_active = false;
        active_goals[i].is_achieved = false;
    }
}

void GoalGenerator::init(uint32_t seed) {
    rng.seed(seed);
    n_active_goals = 0;
    current_goal_idx = -1;
    goal_history_embedding.assign(EMBED_DIM, 0.0f);
    achievement_memory.assign(EMBED_DIM, 0.0f);
    goal_satisfaction = 0.0f;
    goal_drive = 0.3f;
    exploration_curiosity = 0.5f;
    persistence = 0.5f;
    consecutive_failures = 0;
    step_counter = 0;
    achieved_goal_embeddings.clear();
    for (int i = 0; i < MAX_GOALS; i++) {
        active_goals[i].is_active = false;
        active_goals[i].is_achieved = false;
        active_goals[i].pursue_count = 0;
    }
}

void GoalGenerator::generate_candidates(
    const std::vector<float>& self_state,
    const std::vector<float>& world_state,
    const std::vector<float>& concept_activities,
    const std::vector<float>& temporal_context,
    float novelty_drive,
    float competence_drive,
    float self_consistency_drive,
    float curiosity)
{
    step_counter++;

    int max_new = 3;
    int generated = 0;
    for (int i = 0; i < MAX_GOALS; i++) {
        if (!active_goals[i].is_active && generated < max_new) {
            auto& goal = active_goals[i];

            float seed = (float)((generated * 1103515245 + (int)step_counter * 25214903917ULL) % 10000) / 10000.0f;
            int mode = (int)(seed * 3) % 3;

            goal.goal_state.assign(GOAL_DIM, 0.0f);
            for (int d = 0; d < GOAL_DIM; d++) {
                float sval = d < (int)self_state.size() ? self_state[d] : 0.0f;
                float wval = d < (int)world_state.size() ? world_state[d] : 0.0f;
                float cval = d < (int)concept_activities.size() ? concept_activities[d] : 0.0f;
                float tval = d < (int)temporal_context.size() ? temporal_context[d] : 0.0f;

                float noise = (float)((rng() % 2000) - 1000) / 1000.0f * 0.15f;

                if (mode == 0) {
                    goal.goal_state[d] = std::tanh(sval * 1.2f + wval * 0.8f + noise) * 0.5f + 0.5f;
                } else if (mode == 1) {
                    goal.goal_state[d] = std::tanh(cval * 1.5f + tval * 0.5f + noise) * 0.5f + 0.5f;
                } else {
                    goal.goal_state[d] = std::tanh(wval * 1.3f + sval * 0.7f - cval * 0.3f + noise) * 0.5f + 0.5f;
                }
            }

            goal.goal_embedding.assign(EMBED_DIM, 0.0f);
            for (int d = 0; d < EMBED_DIM; d++) {
                goal.goal_embedding[d] = d < GOAL_DIM ? goal.goal_state[d] : 0.0f;
            }
            _normalize(goal.goal_embedding);

            goal.value = 0.0f;
            goal.progress = 0.0f;
            goal.difficulty_estimate = 0.3f + (1.0f - competence_drive) * 0.4f;
            goal.novelty = mode == 2 ? 0.7f : (0.3f + curiosity * 0.4f);
            goal.competence_affordance = competence_drive * 0.5f + 0.3f;
            goal.self_consistency = 0.4f + self_consistency_drive * 0.4f;
            goal.timescale = (int)(seed * 3) % 3;
            goal.created_at = step_counter;
            goal.last_pursued = 0;
            goal.pursue_count = 0;
            goal.is_active = true;
            goal.is_achieved = false;

            generated++;
            n_active_goals = std::max(n_active_goals, i + 1);
        }
    }
}

void GoalGenerator::evaluate_goals(
    const std::vector<float>& current_state,
    float reward_signal,
    float world_change)
{
    exploration_curiosity = exploration_curiosity * 0.95f + world_change * 0.05f;

    for (int i = 0; i < n_active_goals; i++) {
        if (!active_goals[i].is_active || active_goals[i].is_achieved) continue;

        auto& goal = active_goals[i];

        float sim = 0.0f;
        float dist = 0.0f;
        int count = 0;
        for (int d = 0; d < GOAL_DIM && d < (int)current_state.size(); d++) {
            float diff = goal.goal_state[d] - current_state[d];
            dist += diff * diff;
            sim += (1.0f - std::abs(diff));
            count++;
        }
        if (count > 0) {
            dist = std::sqrt(dist / (float)count);
            sim = sim / (float)count;
        }

        goal.progress = 1.0f - std::min(1.0f, dist);

        float time_alive = (float)(step_counter - goal.created_at) / 500.0f;
        float age_penalty = std::exp(-time_alive * 0.3f);

        goal.value = goal.progress * 0.25f
            + goal.competence_affordance * 0.20f
            + goal.novelty * age_penalty * 0.20f
            + goal.self_consistency * 0.15f
            + reward_signal * 0.10f
            + (1.0f - goal.difficulty_estimate) * 0.10f;

        goal.total_desirability = goal.value;
    }
}

void GoalGenerator::select_active_goal(
    const std::vector<float>& self_state,
    float dopamine,
    float meta_conf)
{
    int best = _find_best_goal(self_state);

    if (best >= 0 && best != current_goal_idx) {
        if (current_goal_idx >= 0 && current_goal_idx < n_active_goals) {
            active_goals[current_goal_idx].is_active = false;
        }
        current_goal_idx = best;
    }

    if (current_goal_idx >= 0 && current_goal_idx < n_active_goals) {
        auto& goal = active_goals[current_goal_idx];
        goal.last_pursued = step_counter;
        goal.pursue_count++;
        goal_drive = std::min(1.0f,
            goal_drive + dopamine * 0.1f + meta_conf * goal.value * 0.1f);
    }
}

void GoalGenerator::update_progress(
    const std::vector<float>& current_state,
    const std::vector<float>& target_state)
{
    if (current_goal_idx < 0 || current_goal_idx >= n_active_goals) return;
    auto& goal = active_goals[current_goal_idx];
    if (!goal.is_active || goal.is_achieved) return;

    float dist = 0.0f;
    int count = 0;
    for (int d = 0; d < GOAL_DIM && d < (int)current_state.size(); d++) {
        float diff = goal.goal_state[d] - current_state[d];
        dist += diff * diff;
        count++;
    }
    if (count > 0) dist = std::sqrt(dist / (float)count);

    goal.progress = std::max(0.0f, 1.0f - dist);

    persistence = persistence * 0.9f + (goal.progress > 0.5f ? 0.1f : 0.0f);

    if (goal.progress > 0.9f) {
        mark_achieved(current_goal_idx, goal.progress);
    } else if (dist > 0.8f && goal.pursue_count > 3) {
        consecutive_failures++;
        if (consecutive_failures > 5) {
            goal.is_active = false;
            consecutive_failures = 0;
            current_goal_idx = -1;
        }
    } else {
        consecutive_failures = 0;
    }
}

void GoalGenerator::mark_achieved(int goal_idx, float satisfaction) {
    if (goal_idx < 0 || goal_idx >= MAX_GOALS) return;
    auto& goal = active_goals[goal_idx];
    goal.is_achieved = true;
    goal.is_active = false;
    goal.value = satisfaction;
    goal_satisfaction = goal_satisfaction * 0.7f + satisfaction * 0.3f;

    achieved_goal_embeddings.push_back(goal.goal_embedding);
    if ((int)achieved_goal_embeddings.size() > 32)
        achieved_goal_embeddings.pop_front();

    for (int d = 0; d < EMBED_DIM; d++) {
        achievement_memory[d] = achievement_memory[d] * 0.9f
            + goal.goal_embedding[d] * 0.1f;
    }

    goal_drive = std::min(1.0f, goal_drive + 0.1f);

    if (current_goal_idx == goal_idx)
        current_goal_idx = -1;
}

std::vector<float> GoalGenerator::get_active_goal_embedding() const {
    if (current_goal_idx < 0 || current_goal_idx >= MAX_GOALS)
        return std::vector<float>(EMBED_DIM, 0.0f);
    return active_goals[current_goal_idx].goal_embedding;
}

std::vector<float> GoalGenerator::get_goal_direction() const {
    std::vector<float> dir(GOAL_DIM, 0.0f);
    if (current_goal_idx < 0 || current_goal_idx >= MAX_GOALS)
        return dir;

    const auto& goal = active_goals[current_goal_idx];
    for (int d = 0; d < GOAL_DIM; d++) {
        dir[d] = goal.goal_state[d] * goal.value;
    }
    return dir;
}

std::string GoalGenerator::get_goal_description() const {
    if (current_goal_idx < 0 || current_goal_idx >= MAX_GOALS)
        return "exploring";

    const auto& goal = active_goals[current_goal_idx];
    std::ostringstream oss;

    if (goal.timescale == SHORT_TERM) oss << "st_";
    else if (goal.timescale == LONG_TERM) oss << "lt_";
    else oss << "mt_";

    if (goal.progress > 0.7f) oss << "near_";
    else if (goal.progress > 0.3f) oss << "mid_";
    else oss << "far_";

    float gsum = 0.0f;
    for (int d = 0; d < GOAL_DIM; d++) gsum += goal.goal_state[d];

    if (gsum > GOAL_DIM * 0.6f) oss << "seek";
    else if (gsum > GOAL_DIM * 0.4f) oss << "approach";
    else oss << "explore";

    return oss.str();
}

float GoalGenerator::get_goal_progress() const {
    if (current_goal_idx < 0 || current_goal_idx >= MAX_GOALS) return 0.0f;
    return active_goals[current_goal_idx].progress;
}

bool GoalGenerator::has_active_goal() const {
    return current_goal_idx >= 0 && current_goal_idx < MAX_GOALS;
}

std::string GoalGenerator::get_achievement_summary() const {
    std::ostringstream oss;
    oss << (int)achieved_goal_embeddings.size() << "achieved_"
        << n_active_goals << "active_";
    if (has_active_goal()) {
        oss << "pursuing_" << get_goal_description();
    }
    return oss.str();
}

std::vector<float> GoalGenerator::get_goal_context() const {
    std::vector<float> ctx(16, 0.0f);
    if (current_goal_idx < 0 || current_goal_idx >= MAX_GOALS) return ctx;

    const auto& goal = active_goals[current_goal_idx];
    for (int i = 0; i < 16 && i < GOAL_DIM; i++) {
        ctx[i] = goal.goal_state[i] * goal.value;
    }
    return ctx;
}

void GoalGenerator::decay_stale_goals(uint64_t max_age) {
    for (int i = 0; i < n_active_goals; i++) {
        if (!active_goals[i].is_active) continue;
        uint64_t age = step_counter - active_goals[i].created_at;
        if (age > max_age && active_goals[i].pursue_count == 0) {
            active_goals[i].is_active = false;
            if (i == current_goal_idx) current_goal_idx = -1;
        }
    }
}

int GoalGenerator::_find_best_goal(const std::vector<float>& self_state) const {
    int best = -1;
    float best_val = -1e10f;
    for (int i = 0; i < n_active_goals; i++) {
        if (!active_goals[i].is_active || active_goals[i].is_achieved) continue;

        float dist = 0.0f;
        int count = 0;
        for (int d = 0; d < GOAL_DIM && d < (int)self_state.size(); d++) {
            float diff = active_goals[i].goal_state[d] - self_state[d];
            dist += diff * diff;
            count++;
        }
        if (count > 0) dist = std::sqrt(dist / (float)count);

        float score = active_goals[i].value * 2.0f
            + active_goals[i].novelty * 0.5f
            - active_goals[i].difficulty_estimate * 0.3f
            + (1.0f - dist) * 0.4f;

        if (score > best_val) {
            best_val = score;
            best = i;
        }
    }
    return best;
}

int GoalGenerator::_find_stalest_goal() const {
    int stalest = -1;
    uint64_t max_age = 0;
    for (int i = 0; i < n_active_goals; i++) {
        if (!active_goals[i].is_active) continue;
        uint64_t age = step_counter - active_goals[i].last_pursued;
        if (age > max_age) {
            max_age = age;
            stalest = i;
        }
    }
    return stalest;
}

float GoalGenerator::_cosine_sim(
    const std::vector<float>& a, const std::vector<float>& b) const
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(std::max(1e-10f, na * nb));
    return dot / denom;
}

void GoalGenerator::_normalize(std::vector<float>& v) const {
    float norm = 0.0f;
    for (auto x : v) if (!is_bad(x)) norm += x * x;
    if (norm > 1e-10f) {
        norm = std::sqrt(norm);
        for (auto& x : v) x = x / norm * 0.5f;
    }
}