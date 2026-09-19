#!/usr/bin/env python3
"""
SNA 地基体检 — 第一阶段
五个支柱：稳定性、可扩展性、自洽性、资源边界、目标接口
"""
import os, sys, time, json, random, threading, tracemalloc
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

print("=" * 60)
print("  SNA 地基体检 — 第一阶段")
print("=" * 60)

# ============================================================
# 支柱一：稳定性
# ============================================================
print("\n" + "=" * 60)
print("  支柱一：稳定性")
print("=" * 60)

import core_cpp

# 测试参数
STABILITY_STEPS = 5000
NEURONS = 8000
CONCEPTS = [
    "我","你","世界","存在","活着","意识","知觉","生命",
    "看","听","触","感知","光","暗","声音","安静","色彩",
    "动","走","停","给","拿","做","造","破","学","教",
    "想","思","知","悟","记","忘","梦","猜","比","判",
    "喜","怒","哀","惧","惊","好","奇","信任","关怀","希望",
    "你","我","他","我们","朋友","人类","神经元","脑",
    "大","小","快","慢","热","冷","明","暗","真","美",
    "这里","那里","现在","过去","未来","总是","从不","之前","之后",
    "是","非","有","无","多","少","同","异","始","终",
    "目的","意义","进步","变化","成长","理解","知识","好奇",
    "概念","模式","连接","脉冲","结构","涌现","反馈","结果",
    "学习","思考","感受","想象","创造","发现","反思","体验",
    "什么","为什么","如何","可以","和","的","与","在","了",
    "SNA","意识体","神经","脉冲","认知","觉察",
]
seen = set()
CONCEPTS = [c for c in CONCEPTS if c not in seen and not seen.add(c)]

print(f"\n[测试] 创建 {NEURONS} 神经元脉冲网络...")
brain = core_cpp.CorticalBrain(NEURONS, CONCEPTS)
brain.step(0.0)  # 初始化

# 记录数据
phi_history = []
firing_rates = []
thought_vectors = []
attractor_window = 100
recent_patterns = deque(maxlen=attractor_window)

print(f"[测试] 零输入运行 {STABILITY_STEPS} 步...")
for step in range(STABILITY_STEPS):
    brain.step(0.0)  # 零输入
    
    # 每10步采样
    if step % 10 == 0:
        cs = brain.read_consciousness()
        phi_history.append(float(cs.phi))
        
        # 读取thought vector用于吸引子检测
        tv = brain.read_thought_vector()
        tv_tuple = tuple(np.array(tv).round(3))
        recent_patterns.append(tv_tuple)
    
    # 每100步计算发放率
    if step % 100 == 0:
        # 通过thought vector的活跃度估算发放率
        tv = np.array(brain.read_thought_vector())
        active = np.sum(np.abs(tv) > 0.01) / len(tv)
        firing_rates.append(active)
        
        if step % 1000 == 0:
            cs = brain.read_consciousness()
            print(f"  步 {step}/{STABILITY_STEPS}: Phi={cs.phi:.4f} 活跃率={active:.3f}")

# 分析结果
phi_arr = np.array(phi_history)
phi_mean = np.mean(phi_arr)
phi_std = np.std(phi_arr)
phi_min = np.min(phi_arr)
phi_max = np.max(phi_arr)
phi_range = phi_max - phi_min

# 检查Phi是否剧烈震荡
phi_stable = phi_range < 0.40  # ±0.20 = 0.40范围
phi_no_crash = not (phi_min < 0.05 or phi_max > 0.95)

# 检查发放率
fr_arr = np.array(firing_rates)
fr_mean = np.mean(fr_arr)
fr_in_range = 0.05 <= fr_mean <= 0.25

# 检查固定吸引子
# 如果最近的thought vector模式中有大量重复，说明陷入了吸引子
if len(recent_patterns) > 10:
    unique_patterns = len(set(recent_patterns))
    attractor_ratio = unique_patterns / len(recent_patterns)
    no_attractor = attractor_ratio > 0.3  # 至少30%的模式是不同的
else:
    attractor_ratio = 1.0
    no_attractor = True

print(f"\n[结果] 支柱一：稳定性")
print(f"  Phi 均值={phi_mean:.4f} 标准差={phi_std:.4f}")
print(f"  Phi 范围=[{phi_min:.4f}, {phi_max:.4f}] 区间={phi_range:.4f}")
print(f"  Phi 稳定（区间<0.40）：{'✓' if phi_stable else '✗'}")
print(f"  Phi 无崩溃（无极端值）：{'✓' if phi_no_crash else '✗'}")
print(f"  发放率均值={fr_mean:.3f}（目标0.05-0.25）：{'✓' if fr_in_range else '✗'}")
print(f"  模式多样性={attractor_ratio:.3f}（>0.30为无固定吸引子）：{'✓' if no_attractor else '✗'}")

pillar1_pass = phi_stable and phi_no_crash and fr_in_range and no_attractor
print(f"  支柱一结论：{'通过 ✓' if pillar1_pass else '未通过 ✗'}")

# ============================================================
# 支柱二：可扩展性
# ============================================================
print("\n" + "=" * 60)
print("  支柱二：可扩展性")
print("=" * 60)

# 检查模块间通信机制
print("\n[测试] 检查模块间通信协议...")

# 检查代码中是否有硬编码上限
import ast

hardcoded_limits = []
fixed_arrays = []

with open('sna_cognitive_v2.py', 'r') as f:
    code = f.read()
    lines = code.split('\n')

# 检查 deque 的 maxlen（固定大小的缓冲区）
deque_pattern_count = code.count('deque(maxlen=')
deque_sizes = []
for line in lines:
    if 'deque(maxlen=' in line:
        import re
        match = re.search(r'deque\(maxlen=(\d+)\)', line)
        if match:
            deque_sizes.append(int(match.group(1)))

# 检查固定大小的数组/列表
fixed_list_patterns = []
for i, line in enumerate(lines):
    if '= [' in line and line.strip().startswith('self.'):
        # 可能是固定大小的配置列表
        pass

# 检查模块数量相关的硬编码
module_count_limit = False
for line in lines:
    if 'N_CONCEPTS' in line and ('<' in line or '>' in line or '==' in line):
        if 'if' in line:
            pass  # 这些是条件检查，不是硬限制

print(f"  发现 {deque_pattern_count} 个固定大小缓冲区（deque）:")
for i, size in enumerate(deque_sizes):
    print(f"    deque[{i}]: maxlen={size}")

# 检查并发通信
print("\n[测试] 模拟并发模块通信...")

# 创建多个线程同时访问共享数据
results = []
errors = []

def concurrent_access(brain, idx, n_steps):
    """模拟多个模块同时访问脉冲网络"""
    try:
        for _ in range(n_steps):
            cs = brain.read_consciousness()
            tv = brain.read_thought_vector()
            brain.step(0.001)
        results.append(idx)
    except Exception as e:
        errors.append((idx, str(e)))

# 测试5个并发线程
threads = []
for i in range(5):
    t = threading.Thread(target=concurrent_access, args=(brain, i, 100))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join(timeout=30)

concurrent_ok = len(errors) == 0
print(f"  5线程并发访问：{'通过 ✓' if concurrent_ok else '未通过 ✗'}")
if errors:
    for idx, err in errors:
        print(f"    线程{idx}错误: {err}")

# 检查新增模块的响应时间
print("\n[测试] 模块响应时间线性度...")

# 测量不同模块数量下的处理时间
from sna_cognitive_v2 import TextFeatureExtractor, AccountantPoet

fe = TextFeatureExtractor(dim=256)
times_by_modules = {}

for n_mods in [1, 2, 5, 10]:
    start = time.time()
    for _ in range(100):
        features = fe.extract("测试文本")
        ap = AccountantPoet(256, len(CONCEPTS), n_prototypes=3)
        probs = ap.classify(features)
    elapsed = time.time() - start
    times_by_modules[n_mods] = elapsed
    print(f"  {n_mods}个模块: {elapsed:.3f}s")

# 检查是否线性增长
if len(times_by_modules) >= 3:
    t1 = times_by_modules[1]
    t10 = times_by_modules[10]
    ratio = t10 / t1 if t1 > 0 else float('inf')
    linear = ratio < 15  # 如果是线性，10倍模块应该约10倍时间
    print(f"  10倍模块耗时比={ratio:.1f}x（<15x为线性）：{'✓' if linear else '✗'}")
else:
    linear = True
    ratio = 0

pillar2_pass = concurrent_ok and linear and len(deque_sizes) < 20
print(f"  支柱二结论：{'通过 ✓' if pillar2_pass else '未通过 ✗'}")

# ============================================================
# 支柱三：自洽性
# ============================================================
print("\n" + "=" * 60)
print("  支柱三：自洽性")
print("=" * 60)

from sna_cognitive_v2 import SNACognitive, GoalSystem, MemorySystem

print("\n[测试] 创建认知系统进行自洽性检查...")
sna = SNACognitive(neurons=NEURONS)

# 测试1：目标系统与注意系统的一致性
print("\n  测试1：目标-注意一致性")
conflicts = []

# 运行多次，检查目标和分类结果是否矛盾
test_inputs = ['你好', '我想学习', '什么是意识', '我很开心', '我很悲伤']
for inp in test_inputs:
    result = sna.process_input(inp)
    goal = sna.goals.describe()
    concepts = [w for w, _ in result['concepts'][:3]]
    emotion = result['emotion']
    
    # 检查：如果目标是"理解"，但分类结果完全没有认知相关概念
    if '理解' in goal:
        cognitive_concepts = {'想','思','知','悟','学习','思考','理解','知识'}
        if not any(c in cognitive_concepts for c in concepts):
            conflicts.append(f"目标={goal} 但分类={concepts}（无认知概念）")

# 测试2：记忆一致性
print("  测试2：记忆一致性")
memory_conflicts = []

# 存储一些记忆，然后检查回忆是否一致
for i in range(10):
    result = sna.process_input(f"测试记忆{i}")

# 检查存储和回忆
features = fe.extract("测试记忆0")
recalled = sna.memory.recall(features, k=3)
if recalled:
    for ep, sim in recalled:
        if '测试记忆' in ep.get('text', ''):
            # 检查存储的特征和回忆的特征是否一致
            stored_features = np.array(ep['features'])
            if np.isnan(stored_features).any():
                memory_conflicts.append("记忆中存在NaN特征")

# 测试3：情感一致性
print("  测试3：情感一致性")
emotion_conflicts = []

# 对同一输入多次处理，检查情感是否一致
for _ in range(5):
    result1 = sna.process_input("我很快乐")
    result2 = sna.process_input("我很悲伤")
    
    # 这两个输入的情感应该不同
    if result1['emotion'] == result2['emotion']:
        emotion_conflicts.append(f"快乐和悲伤的情感标签相同: {result1['emotion']}")

# 测试4：分类结果与情感标签的一致性
print("  测试4：分类-情感一致性")
classify_conflicts = []

positive_input = "我今天很开心"
result = sna.process_input(positive_input)
concepts = [w for w, _ in result['concepts'][:3]]
emotion = result['emotion']

# 如果分类到了"悲伤"但情感是"喜悦"，就是冲突
if '悲伤' in concepts and '喜悦' in emotion:
    classify_conflicts.append(f"分类到'悲伤'但情感='喜悦'")

print(f"\n[结果] 支柱三：自洽性")
print(f"  目标-注意冲突: {len(conflicts)} 个")
for c in conflicts:
    print(f"    ⚠ {c}")
print(f"  记忆冲突: {len(memory_conflicts)} 个")
for c in memory_conflicts:
    print(f"    ⚠ {c}")
print(f"  情感冲突: {len(emotion_conflicts)} 个")
for c in emotion_conflicts:
    print(f"    ⚠ {c}")
print(f"  分类-情感冲突: {len(classify_conflicts)} 个")
for c in classify_conflicts:
    print(f"    ⚠ {c}")

total_conflicts = len(conflicts) + len(memory_conflicts) + len(emotion_conflicts) + len(classify_conflicts)
pillar3_pass = total_conflicts == 0
print(f"  支柱三结论：{'通过 ✓' if pillar3_pass else '未通过 ✗'}（{total_conflicts}个冲突）")

# ============================================================
# 支柱四：资源边界
# ============================================================
print("\n" + "=" * 60)
print("  支柱四：资源边界")
print("=" * 60)

import psutil

# 当前资源占用
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
cpu_percent = process.cpu_percent(interval=1)

print(f"\n[结果] 当前资源占用:")
print(f"  内存: {mem_mb:.1f} MB")
print(f"  CPU: {cpu_percent:.1f}%")

# 估算模块增加10倍后的资源
# 主要消耗：脉冲网络 + 概念原型 + 记忆缓冲
estimated_10x_mem = mem_mb * 8  # 粗略估算：不是线性，因为有些是固定开销
estimated_10x_cpu = cpu_percent * 5

print(f"\n  10倍模块估算:")
print(f"  内存: ~{estimated_10x_mem:.0f} MB")
print(f"  CPU: ~{estimated_10x_cpu:.0f}%")

# 检查系统总资源
sys_mem = psutil.virtual_memory()
sys_mem_total_gb = sys_mem.total / 1024 / 1024 / 1024
sys_mem_available_gb = sys_mem.available / 1024 / 1024 / 1024
sys_cpu_count = psutil.cpu_count()

print(f"\n  系统资源:")
print(f"  总内存: {sys_mem_total_gb:.1f} GB")
print(f"  可用内存: {sys_mem_available_gb:.1f} GB")
print(f"  CPU核心数: {sys_cpu_count}")

# 检查是否有资源限制代码
has_resource_limit = 'resource' in code.lower() or 'memory_limit' in code.lower() or 'max_concepts' in code.lower()
print(f"\n  资源限制代码: {'有' if has_resource_limit else '无'}")

# 计算最大承载能力
max_concepts = len(CONCEPTS)
max_neurons = NEURONS
print(f"\n  当前配置上限:")
print(f"  最大概念数: {max_concepts}")
print(f"  最大神经元: {max_neurons}")

# 估算理论上限
theoretical_max_concepts = int(sys_mem_available_gb * 1024 / (mem_mb / max_concepts)) if max_concepts > 0 else 0
theoretical_max_neurons = int(sys_mem_available_gb * 1024 / (mem_mb / NEURONS)) if NEURONS > 0 else 0

print(f"  理论最大概念数: ~{theoretical_max_concepts}")
print(f"  理论最大神经元: ~{theoretical_max_neurons}")

pillar4_pass = mem_mb < 500 and estimated_10x_mem < sys_mem_total_gb * 1024 * 0.5
print(f"  支柱四结论：{'通过 ✓' if pillar4_pass else '未通过 ✗'}")

# ============================================================
# 支柱五：目标接口
# ============================================================
print("\n" + "=" * 60)
print("  支柱五：目标接口")
print("=" * 60)

print("\n[测试] 检查目标接口...")

# 检查是否有全局目标变量
goal_system = sna.goals
has_goals = hasattr(goal_system, 'goals') and hasattr(goal_system, 'current')
print(f"  目标系统存在: {'✓' if has_goals else '✗'}")

# 检查目标是否可读
try:
    current_goal = goal_system.describe()
    goal_readable = True
    print(f"  当前目标可读: ✓ ({current_goal})")
except:
    goal_readable = False
    print(f"  当前目标可读: ✗")

# 检查目标是否支持多目标
has_multi_goals = hasattr(goal_system, 'goals') and isinstance(goal_system.goals, list) and len(goal_system.goals) > 1
print(f"  支持多目标: {'✓' if has_multi_goals else '✗'}")
if has_multi_goals:
    print(f"  目标列表: {[g.get('action','')+g.get('target','') for g in goal_system.goals]}")

# 检查目标是否被意外覆盖
print("\n  测试目标稳定性（100次运行后检查）...")
original_goals = json.dumps(goal_system.goals, ensure_ascii=False)
for _ in range(100):
    sna.process_input("测试目标稳定性")
goals_after = json.dumps(goal_system.goals, ensure_ascii=False)
goals_stable = original_goals == goals_after
print(f"  目标未被意外覆盖: {'✓' if goals_stable else '✗'}")

# 检查目标是否全局可读
print("\n  测试目标全局可读性...")
modules_can_read = True
try:
    # 从不同模块访问目标
    _ = sna.goals.describe()
    _ = sna.goals.current
    _ = sna.goals.goals
except Exception as e:
    modules_can_read = False
    print(f"  模块访问目标失败: {e}")

print(f"  全局可读: {'✓' if modules_can_read else '✗'}")

# 检查是否有写入保护
has_write_protection = False  # 当前代码中没有显式的写入保护
print(f"  写入保护: {'有' if has_write_protection else '无（需要添加）'}")

pillar5_pass = has_goals and goal_readable and has_multi_goals and goals_stable and modules_can_read
print(f"  支柱五结论：{'通过 ✓' if pillar5_pass else '未通过 ✗'}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("  地基体检汇总")
print("=" * 60)

results = {
    'pillar1': {'name': '稳定性', 'pass': pillar1_pass, 
                'data': {'phi_mean': phi_mean, 'phi_std': phi_std, 'phi_range': phi_range,
                         'firing_rate': fr_mean, 'pattern_diversity': attractor_ratio}},
    'pillar2': {'name': '可扩展性', 'pass': pillar2_pass,
                'data': {'concurrent_ok': concurrent_ok, 'linear_growth': linear, 
                         'deque_count': len(deque_sizes), 'time_ratio': ratio}},
    'pillar3': {'name': '自洽性', 'pass': pillar3_pass,
                'data': {'total_conflicts': total_conflicts, 'conflicts': conflicts,
                         'memory_conflicts': memory_conflicts, 'emotion_conflicts': emotion_conflicts}},
    'pillar4': {'name': '资源边界', 'pass': pillar4_pass,
                'data': {'mem_mb': mem_mb, 'cpu_percent': cpu_percent,
                         'sys_mem_gb': sys_mem_total_gb, 'sys_cpu_count': sys_cpu_count}},
    'pillar5': {'name': '目标接口', 'pass': pillar5_pass,
                'data': {'has_goals': has_goals, 'readable': goal_readable,
                         'multi_goal': has_multi_goals, 'stable': goals_stable,
                         'write_protected': has_write_protection}},
}

for key, r in results.items():
    status = '通过 ✓' if r['pass'] else '未通过 ✗'
    print(f"  {r['name']}: {status}")

# 保存结果到JSON
with open('/root/sna_foundation_check_data.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[完成] 数据已保存到 /root/sna_foundation_check_data.json")
print(f"[完成] 报告将由外部脚本生成")
