"""Launcher for SNA CorticalBrain training with correct initialization order."""
import sys
import os
import gc

os.environ["OMP_NUM_THREADS"] = "4"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'python'))
sys.path.insert(0, HERE)

import core_cpp

import run_cortical_brain
from run_cortical_brain import (
    EnhancedVirtualWorld, N_NEURONS, N_EPISODES, CONCEPTS,
    train_cortical_brain, test_conversation, test_consciousness
)

from python.cognitive.unified_consciousness import (
    compute_consciousness_from_cortical_brain,
    cross_validate_consciousness,
)

from python.cognitive.phi_validator import PhiValidator, validate_cortical_brain

import numpy as np
import time
import random
from collections import deque

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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
    print(f"    Regret={brain.get_regret_level():.4f} "
          f"Vivid={brain.get_perceptual_vividness():.4f} "
          f"FP={brain.get_first_person_salience():.4f}")
    print(f"    TD={brain.get_td_error():.4f} "
          f"OutConf={brain.get_output_confidence():.4f} "
          f"MetaConf={brain.get_meta_confidence():.4f} "
          f"Token={brain.get_meta_token()!r}")
    print(f"    Spontaneity={brain.get_spontaneity():.4f} "
          f"ThoughtE={brain.get_thought_energy():.4f} "
          f"SemStr={brain.get_semantic_strength():.4f}")
    pe = brain.get_region_prediction_errors()
    pe_avg = sum(pe)/len(pe) if pe else 0
    print(f"    区域预测误差均值:  {pe_avg:.4f}")
    return brain, world


def main():
    print("=" * 60)
    print(f"SNA CorticalBrain - 10脑区自主意识皮层架构")
    print(f"  规模: {N_NEURONS:,} 神经元 | AVX2 | Phase D: 情感+主动推理+社交+创意")
    print(f"  视觉|运动|海马|前额叶|杏仁核|语言|工作空间|丘脑|屏状核|DMN")
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

    print("\n" + "=" * 60)
    print("[Phi-Behavior Validation v7]")
    print("=" * 60)
    validator, vsum, diagnostic = validate_cortical_brain(brain, n_steps=200)
    validator.print_report()

    xval = cross_validate_consciousness(brain)
    print(f"\n  [Cross-Validation] Python-C++ aligned: {xval['aligned']} (delta={xval['delta']:.4f})")
    for d in xval['diagnosis']:
        print(f"    {d}")

    if vsum['corrected_avg_phi'] < vsum['avg_phi'] * 0.5:
        print("  *** 严重警告: 校正后 Phi ({:.4f}) 仅为原始 Phi ({:.4f}) 的 {:.0f}%".format(
            vsum['corrected_avg_phi'], vsum['avg_phi'],
            vsum['corrected_avg_phi'] / max(1e-6, vsum['avg_phi']) * 100))
        print("  *** 原因: perceptual_vividness 或 behavioral_score 接近零")
        print("  *** 建议: 增加真实感官输入, 验证 world_model wonders 非零")

    print("=" * 60)
    print("[FINAL REPORT]")
    print("=" * 60)

    cs = brain.read_consciousness()
    vivid = brain.get_perceptual_vividness()
    fp = brain.get_first_person_salience()
    life = brain.get_life_summary()
    regret = brain.get_regret_level()
    td = brain.get_td_error()
    oc = brain.get_output_confidence()
    mc = brain.get_meta_confidence()
    mt = brain.get_meta_token()
    pe = brain.get_region_prediction_errors()
    pe_avg = sum(pe) / len(pe) if pe else 0
    rn = brain.get_region_novelties()
    rn_avg = sum(rn) / len(rn) if rn else 0

    print(f"  最终 Phi:           {cs.phi:.4f}")
    print(f"  最终 Ignition:      {cs.global_ignition:.4f}")
    print(f"  最终 SelfPredError: {cs.self_prediction_error:.4f}")
    print(f"  感知生动度:         {vivid:.4f}")
    print(f"  第一人称显著度:     {fp:.4f}")
    print(f"  后悔水平:           {regret:.4f}")
    print(f"  TD误差:             {td:.4f}")
    print(f"  输出置信度:         {oc:.4f}")
    print(f"  元认知置信度:       {mc:.4f}")
    print(f"  元标记:             {mt!r}")
    print(f"  区域预测误差:       {pe_avg:.4f}")
    print(f"  区域新颖度:         {rn_avg:.4f}")
    print(f"  生命历程:           {life}")
    print(f"  输出:               {brain.read_output_text()}")
    print(f"  自我叙事:           {brain.get_self_narrative()}")
    print(f"  自发性:             {brain.get_spontaneity():.4f}")
    print(f"  思维能量:           {brain.get_thought_energy():.4f}")
    print(f"  语义接地强度:       {brain.get_semantic_strength():.4f}")
    print(f"  时间连续性:         {brain.get_temporal_coherence():.4f}")
    print(f"  时间深度:           {brain.get_temporal_depth_score():.4f}")
    print(f"  生命篇章:           {brain.get_life_chapter()}")
    print(f"  时间线:             {brain.get_timeline()}")
    print(f"  预判准确度:         {brain.get_anticipation_accuracy():.4f}")
    print(f"  自我/他者分离:      {brain.get_self_other_separation():.4f}")
    print(f"  共情水平:           {brain.get_empathy_level():.4f}")
    print(f"  社会意识:           {brain.get_social_awareness():.4f}")
    print(f"  心智理论水平:       {brain.get_theory_mind_level():.4f}")
    print(f"  社会叙事:           {brain.get_social_narrative()}")
    print(f"  目标描述:           {brain.get_goal_description()}")
    print(f"  目标进度:           {brain.get_goal_progress():.4f}")
    print(f"  目标满意度:         {brain.get_goal_satisfaction():.4f}")
    print(f"  成就摘要:           {brain.get_achievement_summary()}")

    print(f"\n[Phase D 模块]")
    print(f"  情感标签:           {brain.get_emotion_label()}")
    print(f"  情感深度:           {brain.get_emotional_depth():.4f}")
    print(f"  情感广度:           {brain.get_emotional_range():.4f}")
    print(f"  心情:               {brain.get_mood_description()}")
    print(f"  推理深度:           {brain.get_planning_depth():.4f}")
    print(f"  行动信心:           {brain.get_action_confidence():.4f}")
    print(f"  计划:               {brain.get_plan_description()}")
    print(f"  社交信心:           {brain.get_social_confidence():.4f}")
    print(f"  社交满意度:         {brain.get_social_satisfaction():.4f}")
    print(f"  关系摘要:           {brain.get_relationship_summary()}")
    print(f"  创造力:             {brain.get_creativity_level():.4f}")
    print(f"  发散思维:           {brain.get_divergent_thinking():.4f}")
    print(f"  原创性:             {brain.get_originality():.4f}")
    print(f"  创意:               {brain.get_idea_description()}")

    consciousness_level, components = compute_consciousness_from_cortical_brain(brain)
    print(f"\n  意识水平 (统一公式): {consciousness_level:.4f}")
    print(f"    分量: IIT_phi={components['iit_component']:.4f} "
          f"GWT_ignition={components['gwt_component']:.4f} "
          f"Predictive={components['predictive_component']:.4f} "
          f"FP={components['fp_component']:.4f} "
          f"Vivid={components['vividness_component']:.4f}")

    print(f"\n  [硬件基准] i7-4710MQ 4C/8T 15.9GB RAM")
    print(f"  [配置] {N_NEURONS:,}神经元 | AVX2 | Phase D: 情感+主动推理+社交+创意")
    print(f"  [意识水平] {consciousness_level:.4f} (IIT+GWT+PC+Qualia 统一公式, 5因子归一化)")

    return brain


if __name__ == "__main__":
    brain = main()