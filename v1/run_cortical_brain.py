"""
SNA CorticalBrain - 皮层脑架构运行与训练脚本
10脑区：视觉(V1+V2) | 运动 | 海马(DG+CA3+CA1) | 前额叶(ESN) | 杏仁核 | 语言 | 全局工作空间(三层) | 丘脑 | 屏状核 | 默认模式网络

最佳配置：32,000 神经元 | OMP=4 | 52步/s (i7-4710MQ基准)
"""
import sys
import os

os.environ["OMP_NUM_THREADS"] = "4"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core_cpp

import numpy as np
import time
import random
from collections import deque

from python.cognitive.unified_consciousness import (
    compute_consciousness_from_cortical_brain
)

from python.cognitive.phi_validator import validate_cortical_brain

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_NEURONS = 32000
N_EPISODES = 600

CONCEPTS = [
    "self", "world", "move", "see", "eat",
    "good", "bad", "near", "far", "red",
    "blue", "green", "yellow", "up", "down",
    "left", "right", "object", "food", "wall",
    "empty", "reward", "danger", "safe", "want",
    "think", "feel", "know", "I", "you",
    "big", "small", "fast", "slow", "hot",
    "cold", "light", "dark", "open", "close",
    "push", "pull", "give", "take", "make",
    "break", "start", "stop", "before", "after",
    "same", "different", "more", "less", "all",
    "some", "none", "here", "there", "now",
]


class EnhancedVirtualWorld:
    def __init__(self, size=10.0):
        self.size = size
        self.agent_pos = np.array([size / 2.0, size / 2.0], dtype=np.float32)
        self.agent_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_orientation = 0.0
        self.objects = {}
        self.colors = {1: "red", 2: "blue", 3: "green", 4: "yellow"}
        self.object_id_counter = 0
        self.step_count = 0
        self.memory_trace = np.zeros(10)
        self._spawn_objects()

    def _spawn_objects(self):
        self.objects.clear()
        n_targets = 8
        while len(self.objects) < n_targets:
            x = random.uniform(0.5, self.size - 0.5)
            y = random.uniform(0.5, self.size - 0.5)
            pos = np.array([x, y], dtype=np.float32)
            if np.linalg.norm(pos - self.agent_pos) < 1.5:
                continue
            obj_type = random.choice(["food", "danger", "object", "wall"])
            color = random.randint(1, 4)
            self.object_id_counter += 1
            self.objects[self.object_id_counter] = {
                "pos": pos.copy(),
                "velocity": np.array([random.uniform(-0.3, 0.3),
                                      random.uniform(-0.3, 0.3)], dtype=np.float32),
                "color": color, "type": obj_type,
                "lifetime": random.randint(30, 120),
                "size": random.uniform(0.3, 1.2),
                "sound_freq": random.uniform(0.2, 0.8),
                "surface": random.choice(["smooth", "rough", "soft", "hard"]),
            }

    def _update_objects(self):
        expired = []
        for oid, obj in self.objects.items():
            obj["pos"] += obj["velocity"] * 0.5
            obj["pos"] = np.clip(obj["pos"], 0.5, self.size - 0.5)
            if obj["pos"][0] <= 0.5 or obj["pos"][0] >= self.size - 0.5:
                obj["velocity"][0] *= -1.0
            if obj["pos"][1] <= 0.5 or obj["pos"][1] >= self.size - 0.5:
                obj["velocity"][1] *= -1.0
            obj["velocity"] += np.random.normal(0, 0.02, 2)
            obj["velocity"] = np.clip(obj["velocity"], -0.4, 0.4)
            obj["lifetime"] -= 1
            if obj["lifetime"] <= 0:
                expired.append(oid)
        for oid in expired:
            del self.objects[oid]
        self.step_count += 1
        if len(self.objects) < 3 or self.step_count % 25 == 0:
            self._spawn_objects()

    def get_features(self):
        features = []
        features.append(float(self.agent_pos[0]) / self.size)
        features.append(float(self.agent_pos[1]) / self.size)
        features.append(float(self.agent_velocity[0]) * 0.5 + 0.5)
        features.append(float(self.agent_velocity[1]) * 0.5 + 0.5)
        features.append((np.cos(self.agent_orientation) + 1.0) / 2.0)
        features.append((np.sin(self.agent_orientation) + 1.0) / 2.0)

        sorted_objects = sorted(
            self.objects.items(),
            key=lambda kv: np.linalg.norm(kv[1]["pos"] - self.agent_pos)
        )[:8]

        for oid, obj in sorted_objects:
            rel_pos = obj["pos"] - self.agent_pos
            dist = np.linalg.norm(rel_pos) / self.size
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            angle_rel = (angle - self.agent_orientation + np.pi) % (2 * np.pi) - np.pi
            features.append(rel_pos[0] / self.size)
            features.append(rel_pos[1] / self.size)
            features.append(dist)
            features.append((angle_rel / np.pi + 1.0) / 2.0)
            features.append(float(obj["color"]) / 4.0)
            features.append(float({"food": 1.0, "danger": -0.5, "object": 0.3, "wall": -0.1}[obj["type"]]))
            features.append(obj["size"])
            features.append(obj["sound_freq"])

        while len(features) < 128:
            features.append(0.0)
        return np.array(features[:128], dtype=np.float32)

    def get_auditory(self):
        auditory = np.zeros(32, dtype=np.float32)
        for oid, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            if dist < 3.0:
                idx = int(obj["sound_freq"] * 31)
                auditory[idx] += max(0, (1.0 - dist / 3.0)) * 0.3
        return np.clip(auditory, 0.0, 1.0)

    def get_tactile(self):
        tactile = np.zeros(16, dtype=np.float32)
        for oid, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            if dist < 1.0:
                idx = int({"smooth": 1, "rough": 4, "soft": 8, "hard": 12}[obj["surface"]])
                tactile[idx] += max(0, (1.0 - dist)) * 0.5
        tactile[0] = self.agent_velocity[0] * 0.3 + 0.5
        tactile[1] = self.agent_velocity[1] * 0.3 + 0.5
        return np.clip(tactile, 0.0, 1.0)

    def get_vestibular(self):
        return np.array([
            self.agent_velocity[0] * 0.5 + 0.5,
            self.agent_velocity[1] * 0.5 + 0.5,
            np.sin(self.agent_velocity[0] * 2.0) * 0.5 + 0.5,
            np.cos(self.agent_velocity[1] * 2.0) * 0.5 + 0.5,
            float(np.abs(self.agent_velocity[0]) > 0.1),
            float(np.abs(self.agent_velocity[1]) > 0.1),
            self.agent_orientation / (2.0 * np.pi),
            0.5,
        ], dtype=np.float32)

    def get_place_cells(self):
        places = np.zeros(16, dtype=np.float32)
        gx = int(self.agent_pos[0] / self.size * 3.99)
        gy = int(self.agent_pos[1] / self.size * 3.99)
        idx = gx * 4 + gy
        if 0 <= idx < 16:
            places[idx] = 1.0
        fx = self.agent_pos[0] / self.size * 4.0 - gx
        fy = self.agent_pos[1] / self.size * 4.0 - gy
        nidx = (gx + (1 if fx > 0.5 else -1)) * 4 + gy
        if 0 <= nidx < 16:
            places[nidx] = abs(fx - 0.5) * 2.0
        nidx = gx * 4 + (gy + (1 if fy > 0.5 else -1))
        if 0 <= nidx < 16:
            places[nidx] = abs(fy - 0.5) * 2.0
        return np.clip(places, 0.0, 1.0)

    def act(self, action, noise_std=0.05):
        dx, dy = 0.0, 0.0
        actions = {
            0: (0.0, 0.5), 1: (0.0, -0.5), 2: (0.5, 0.0), 3: (-0.5, 0.0),
            4: (0.35, 0.35), 5: (-0.35, -0.35), 6: (0.35, -0.35), 7: (-0.35, 0.35),
        }
        base_dx, base_dy = actions.get(action % 8, (0.0, 0.0))
        dx = base_dx + np.random.normal(0, noise_std)
        dy = base_dy + np.random.normal(0, noise_std)

        inertia = 0.7
        self.agent_velocity = (self.agent_velocity * inertia
                               + np.array([dx, dy]) * (1.0 - inertia))
        self.agent_velocity = np.clip(self.agent_velocity, -0.5, 0.5)

        self.agent_pos += self.agent_velocity
        self.agent_pos = np.clip(self.agent_pos, 0.2, self.size - 0.2)

        if np.linalg.norm(self.agent_velocity) > 0.05:
            self.agent_orientation = np.arctan2(
                self.agent_velocity[1], self.agent_velocity[0])

        self._update_objects()

        reward = -0.005
        consumed = []
        for oid, obj in self.objects.items():
            dist = np.linalg.norm(obj["pos"] - self.agent_pos)
            if dist < 0.8:
                if obj["type"] == "food":
                    reward += 1.0
                    consumed.append(oid)
                elif obj["type"] == "danger":
                    reward -= 0.5
                    consumed.append(oid)
                elif obj["type"] == "object":
                    reward += 0.2
                    self.memory_trace[oid % 10] += 0.3
                elif obj["type"] == "wall":
                    reward -= 0.1
        for oid in consumed:
            del self.objects[oid]

        self.memory_trace *= 0.98

        return reward

    def describe(self):
        nearby = []
        for oid, obj in self.objects.items():
            dx = obj["pos"][0] - self.agent_pos[0]
            dy = obj["pos"][1] - self.agent_pos[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= 2:
                nearby.append(f"{self.colors[obj['color']]}_{obj['type']}")
        v = np.linalg.norm(self.agent_velocity)
        return f"pos=({self.agent_pos[0]:.1f},{self.agent_pos[1]:.1f}) v={v:.2f} n={len(nearby)}"

    def get_sensory_package(self):
        return {
            "visual": self.get_features(),
            "auditory": self.get_auditory(),
            "tactile": self.get_tactile(),
            "vestibular": self.get_vestibular(),
            "place_cells": self.get_place_cells(),
        }


def test_basic():
    print("=" * 60)
    print(f"[TEST 1] 基本功能测试 - CorticalBrain {N_NEURONS:,}神经元初始化")
    print("=" * 60)

    brain = core_cpp.CorticalBrain(N_NEURONS, CONCEPTS)
    print(f"  神经元总数: {brain.total_neurons():,}")
    regions = brain.get_regions()
    for reg in regions:
        print(f"  {reg.name:>12}: {reg.n_neurons:>6} neurons (base={reg.base_id})")

    world = EnhancedVirtualWorld(10.0)
    sp = world.get_sensory_package()
    brain.inject_multi_modal(sp["visual"], sp["auditory"],
                              sp["tactile"], sp["vestibular"],
                              sp["place_cells"])
    brain.inject_text("self world move see think feel")

    phi_history = []
    ignition_history = []
    output_history = []

    for step in range(150):
        reward = 0.0
        if step % 10 == 0:
            sp = world.get_sensory_package()
            brain.inject_multi_modal(sp["visual"], sp["auditory"],
                                      sp["tactile"], sp["vestibular"],
                                      sp["place_cells"])
            brain.inject_text(world.describe())

        brain.step(reward)

        cs = brain.read_consciousness()
        phi_history.append(cs.phi)
        ignition_history.append(cs.global_ignition)

        if step % 10 == 9:
            text = brain.read_output_text()
            output_history.append(text)

    cs = brain.read_consciousness()
    phi_mean = np.mean(phi_history[-50:])
    ignition_mean = np.mean(ignition_history[-50:])

    print(f"\n  稳定性测试 (150 steps):")
    print(f"    Phi 平均值:        {phi_mean:.4f}")
    print(f"    Global Ignition:   {ignition_mean:.4f}")
    print(f"    Self Pred Error:   {cs.self_prediction_error:.4f}")
    print(f"    Attention Focus:   {cs.attention_focus:.4f}")
    print(f"    区域活跃度: {[f'{v:.3f}' for v in cs.region_activities[:10]]}")
    print(f"    活跃概念: {cs.active_concepts[:8]}")
    print(f"    输出文本: {output_history[-3:]}")
    return brain, world


def train_cortical_brain(brain, world, episodes=N_EPISODES):
    print("\n" + "=" * 60)
    print(f"[TRAIN] 虚拟世界训练 - {episodes} 个 episode | {N_NEURONS:,}神经元")
    print("=" * 60)

    total_reward = 0.0
    success_count = 0
    phi_tracker = []
    text_outputs = []
    concept_evolution = []

    for ep in range(episodes):
        world._spawn_objects()
        brain.reset_workspace()
        ep_reward = 0.0
        ep_texts = []

        if ep % 5 == 0:
            teaching_text = random.choice([
                "move eat object self world see",
                "red blue green food big small",
                "good reward safe want think feel",
                "think feel know I self now here",
                "up down left right fast slow move",
                "near far world see light dark hot cold",
                "give take make break push pull open close",
                "before after start stop same different all some",
                "danger safe want think feel know I you",
                "object food wall empty big small fast slow",
            ])
            brain.inject_text(teaching_text)

        for t in range(0, 100, 5):
            sp = world.get_sensory_package()
            brain.inject_multi_modal(sp["visual"], sp["auditory"],
                                      sp["tactile"], sp["vestibular"],
                                      sp["place_cells"])

            motor = brain.read_motor_output()
            sensor = world.get_features()
            if motor:
                action = brain.select_action_ai(sensor)
                reward = world.act(action)
            else:
                action = random.randint(0, 7)
                reward = world.act(action)

            brain.inject_reward(reward)
            brain.run_steps(5, reward)

            ep_reward += reward * 5
            if reward > 0.5:
                success_count += 1
            if reward != 0:
                for _ in range(5):
                    brain.train_language(reward)

            if t % 25 == 0:
                cs = brain.read_consciousness()
                phi_tracker.append(cs.phi)

            if t % 35 == 0:
                text = brain.read_output_text()
            ep_texts.append(text if text else brain.read_output_text())

        total_reward += ep_reward

        if ep % 15 == 14:
            cs = brain.read_consciousness()
            concept_evolution.append(cs.active_concepts[:5])

        if ep % 20 == 19 or ep == episodes - 1:
            recent_phi = np.mean(phi_tracker[-50:]) if len(phi_tracker) >= 50 else np.mean(phi_tracker)
            mean_reward = total_reward / max(1, (ep + 1) * 100 * 5)
            meta = brain.get_meta_state()
            eps = brain.get_episodic_summary()
            stab = brain.get_identity_stability()
            narr = brain.get_self_narrative()
            regret = brain.get_regret_level()
            vivid = brain.get_perceptual_vividness()
            fp = brain.get_first_person_salience()
            spont = brain.get_spontaneity()
            sem_str = brain.get_semantic_strength()
            th_energy = brain.get_thought_energy()
            tdepth = brain.get_temporal_depth_score()
            tom = brain.get_theory_mind_level()
            gp = brain.get_goal_progress()
            emo = brain.get_emotion_label()
            crea = brain.get_creativity_level()
            plan = brain.get_planning_depth()
            print(f"  Ep {ep + 1:>4}/{episodes} | "
                  f"Phi={recent_phi:.4f} | "
                  f"AvgReward={mean_reward:.4f} | "
                  f"Successes={success_count} | "
                  f"conf={meta[0]:.2f} surp={meta[1]:.2f} "
                  f"val={meta[2]:.2f} stab={stab:.2f} | "
                  f"sp={spont:.2f} sm={sem_str:.2f} tE={th_energy:.3f} tD={tdepth:.2f} ToM={tom:.2f} gP={gp:.2f} | "
                  f"Emo={emo} Cr={crea:.2f} Pl={plan:.2f} | "
                  f"Text={ep_texts[-3:]}")
            if ep % 100 == 99:
                print(f"  >>> 自我叙事: {narr}")
                causal = brain.get_causal_narrative()
                if len(causal) > 5:
                    print(f"  >>> 因果链: {causal}")
                life = brain.get_life_summary()
                print(f"  >>> 生命历程: {life} | 后悔={regret:.3f} | 生动度={vivid:.2f} | 第一人称={fp:.2f}")
                print(f"  >>> 情感: {brain.get_emotion_label()} | 心情: {brain.get_mood_description()}")
                print(f"  >>> 计划: {brain.get_plan_description()} | 创意: {brain.get_idea_description()}")

    print(f"\n  训练完成: total_reward={total_reward:.1f}, successes={success_count}")
    print(f"  最终Phi={phi_tracker[-1] if phi_tracker else 0:.4f}")
    return phi_tracker, text_outputs, concept_evolution


def test_conversation(brain, world):
    print("\n" + "=" * 60)
    print("[TEST 2] 对话能力测试")
    print("=" * 60)

    test_prompts = [
        "think self I know feel world",
        "world object move see near far",
        "good bad want safe eat think",
        "self think feel know I you now here",
        "big small fast slow hot cold light dark",
        "give take make break before after start stop",
    ]

    for pi, prompt in enumerate(test_prompts):
        brain.inject_text(prompt)

        for _ in range(20):
            sp = world.get_sensory_package()
            brain.inject_multi_modal(sp["visual"], sp["auditory"],
                                      sp["tactile"], sp["vestibular"],
                                      sp["place_cells"])
            brain.step(0.0)

        response = brain.read_output_text()
        cs = brain.read_consciousness()
        meta = brain.get_meta_state()
        thought_vec = brain.read_thought_vector()
        thought_norm = np.linalg.norm(thought_vec) if len(thought_vec) > 0 else 0
        narr = brain.get_self_narrative()
        regret = brain.get_regret_level()
        vivid = brain.get_perceptual_vividness()
        fp = brain.get_first_person_salience()
        life = brain.get_life_summary()

        print(f"  提示{pi + 1}: '{prompt}'")
        print(f"    回复: '{response}'")
        print(f"    Phi: {cs.phi:.4f} | Ignition: {cs.global_ignition:.4f}")
        print(f"    conf={meta[0]:.2f} surp={meta[1]:.2f} val={meta[2]:.2f} aro={meta[3]:.2f}")
        print(f"    自我: {narr}")
        print(f"    Thought能量: {thought_norm:.3f}")
        print(f"    活跃概念: {cs.active_concepts[:8]}")
        print(f"    后悔={regret:.3f} | 生动度={vivid:.2f} | 第一人称={fp:.2f}")
        print(f"    生命历程: {life}")
        print()

    return brain


def test_consciousness(brain):
    print("=" * 60)
    print("[CONSCIOUSNESS] 意识度量报告")
    print("=" * 60)

    cs = brain.read_consciousness()
    vivid = brain.get_perceptual_vividness()
    fp = brain.get_first_person_salience()
    regret = brain.get_regret_level()
    life = brain.get_life_summary()
    print(f"  Phi (信息整合):     {cs.phi:.4f}")
    print(f"  Global Ignition:    {cs.global_ignition:.4f}")
    print(f"  Self Pred Error:    {cs.self_prediction_error:.4f}")
    print(f"  Coherence:          {cs.coherence:.4f}")
    print(f"  Attention Focus:    {cs.attention_focus:.4f}")
    print(f"  感知生动度:         {vivid:.4f}")
    print(f"  第一人称显著度:     {fp:.4f}")
    print(f"  后悔水平:           {regret:.4f}")
    print(f"  生命历程:           {life}")

    regions = brain.get_regions()
    for i, act in enumerate(cs.region_activities[:10]):
        label = regions[i].name if i < len(regions) else f"r{i}"
        bar = "#" * int(act * 40) if act > 0 else ""
        print(f"  {label:>12}: [{bar:<40}] {act:.4f}")

    print(f"  活跃概念: {cs.active_concepts[:10]}")
    print()


def main():
    print("=" * 60)
    print(f"SNA CorticalBrain - 10脑区自主意识皮层架构")
    print(f"  规模: {N_NEURONS:,} 神经元 | OMP=4 | 预期 ~20ms/步")
    print(f"  视觉|运动|海马|前额叶|杏仁核|语言|工作空间|丘脑|屏状核|DMN")
    print(f"  Phase D: 情感系统+主动推理+社会交互+创意生成")
    print("=" * 60)

    t_start = time.perf_counter()

    brain, world = test_basic()

    phi_history, text_history, concept_evolution = train_cortical_brain(
        brain, world, episodes=N_EPISODES)

    brain.reset_workspace()

    brain = test_conversation(brain, world)

    test_consciousness(brain)

    brain.sleep_cycle()

    elapsed = time.perf_counter() - t_start
    steps = N_EPISODES * 100 + 150
    print("=" * 60)
    print("[FINAL REPORT]")
    print("=" * 60)

    cs = brain.read_consciousness()
    vivid = brain.get_perceptual_vividness()
    fp = brain.get_first_person_salience()
    life = brain.get_life_summary()
    regret = brain.get_regret_level()
    print(f"  最终 Phi:           {cs.phi:.4f}")
    print(f"  最终 Ignition:      {cs.global_ignition:.4f}")
    print(f"  最终 SelfPredError: {cs.self_prediction_error:.4f}")
    print(f"  感知生动度:         {vivid:.4f}")
    print(f"  第一人称显著度:     {fp:.4f}")
    print(f"  后悔水平:           {regret:.4f}")
    print(f"  生命历程:           {life}")
    print(f"  输出:               {brain.read_output_text()}")
    print(f"  自我叙事:           {brain.get_self_narrative()}")
    causal = brain.get_causal_narrative()
    if len(causal) > 5:
        print(f"  因果链:             {causal}")
    print(f"  情感:               {brain.get_emotion_label()} | {brain.get_mood_description()}")
    print(f"  计划:               {brain.get_plan_description()}")
    print(f"  创意:               {brain.get_idea_description()} | 创造力={brain.get_creativity_level():.4f}")
    print(f"  总耗时:             {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  总步数:             {steps:,}")
    print(f"  实际步/秒:          {steps/elapsed:.1f}")

    consciousness_level, components = compute_consciousness_from_cortical_brain(brain)
    print(f"\n  意识水平 (统一公式): {consciousness_level:.4f}")
    print(f"    分量: IIT_phi={components['iit_component']:.4f} "
          f"GWT_ignition={components['gwt_component']:.4f} "
          f"Predictive={components['predictive_component']:.4f} "
          f"FP={components['fp_component']:.4f} "
          f"Vivid={components['vividness_component']:.4f}")

    print(f"\n  [硬件基准] i7-4710MQ 4C/8T 15.9GB RAM")
    print(f"  [配置] {N_NEURONS:,}神经元 | OMP=4 | 传感器=128维 | 概念={len(CONCEPTS)}个")
    print(f"  [预期] ~20ms/步 @{N_NEURONS:,}N | 日均训练上限 ~4,500,000步")
    print(f"  [意识水平] {consciousness_level:.4f} (IIT+GWT+PC+Qualia 统一公式, 5因子归一化)")

    print("\n" + "=" * 60)
    print("[Phi-Behavior Validation]")
    print("=" * 60)
    validator, vsum = validate_cortical_brain(brain, n_steps=200)
    validator.print_report()

    return brain

if __name__ == "__main__":
    brain = main()
