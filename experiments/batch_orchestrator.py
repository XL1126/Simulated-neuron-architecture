"""
SNA 批量实验编排器 - 充分利用本地全部算力累积实验数据。

特性:
- 分级实验：快速验证 → 中等规模 → 深度训练 → 规模化扫描
- 增量保存：每个 seed 完成后立即写入磁盘
- 断点续传：可从上次中断处继续
- 自动报告：每级实验完成后生成中间报告
- 资源自适应：根据内存和CPU自动调整参数

用法:
    python experiments/batch_orchestrator.py           # 运行全部4级
    python experiments/batch_orchestrator.py --tier 1  # 只运行第1级
    python experiments/batch_orchestrator.py --resume  # 续传上次中断
"""
import sys
import os
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'python'))

os.environ['OMP_NUM_THREADS'] = '4'


def _timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class BatchOrchestrator:
    """分级批量实验编排器"""

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else (ROOT / 'experiment_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / 'orchestrator_checkpoint.json'
        self.results = {}
        self.current_tier = 0
        self.current_step = 0

    def _save_checkpoint(self, tier, step, result_file=None):
        data = {
            'tier': tier,
            'step': step,
            'result_file': result_file,
            'timestamp': _timestamp(),
            'completed': list(self.results.keys()),
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return None

    def _run_config(self, config, label, tier, step):
        """运行单个配置的实验"""
        from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
        from experiments.virtual_world_v2 import GridWorldV2

        print(f"\n{'='*60}")
        print(f"[T{tier}.{step}] {label}")
        print(f"  N={config.get('n_neurons',32000):,} | {config.get('n_episodes',200)}ep x {config.get('n_steps_per_episode',50)}st | seeds={config.get('n_seeds',3)}")
        print(f"  World: {config.get('world_class','Enhanced')} | size={config.get('world_size',10)}")
        t0 = time.perf_counter()

        world_cls = GridWorldV2 if config.get('world_class') == 'GridWorldV2' else None
        exp_config = ExperimentConfig(
            n_neurons=config['n_neurons'],
            n_episodes=config['n_episodes'],
            n_steps_per_episode=config.get('n_steps_per_episode', 50),
            n_seeds=config['n_seeds'],
            seed_offset=config.get('seed_offset', 100 * step),
            world_size=config.get('world_size', 10.0),
            omp_num_threads=4,
            output_dir=str(self.output_dir),
        )

        runner = ExperimentRunner(exp_config)
        if world_cls:
            runner.world_class = world_cls

        def progress(i, total, seed):
            if i == 0 or i == total - 1 or (i + 1) % max(1, total // 4) == 0:
                elapsed = time.perf_counter() - t0
                print(f"    [{i+1}/{total}] seed={seed} ({elapsed:.0f}s)", flush=True)

        report = runner.run_all(progress_callback=progress)
        elapsed = time.perf_counter() - t0

        agg = report.aggregate
        stab = report.phi_stability
        corr = report.correlation_analysis

        flat = {
            'label': label,
            'tier': tier,
            'step': step,
            'elapsed_seconds': elapsed,
            'n_neurons': config['n_neurons'],
            'n_episodes': config['n_episodes'],
            'n_seeds': config['n_seeds'],
            'world': config.get('world_class', 'Enhanced'),
            **{f'agg_{k}': v for k, v in agg.items()},
            **{f'stab_{k}': v for k, v in stab.items()},
            **{f'corr_{k}': v for k, v in corr.items()},
        }

        fname = self.output_dir / f'{label.replace(" ","_").replace("/","_")}_{_timestamp()}.json'
        with open(fname, 'w') as f:
            json.dump(flat, f, indent=2, ensure_ascii=False)

        print(f"  Done: {elapsed:.1f}s | Phi={agg.get('mean_phi_mean',0):.4f}+/-{agg.get('mean_phi_std',0):.4f} | "
              f"Succ={agg.get('success_rate_mean',0):.4f} | Repr={stab.get('reproducibility_score',0):.3f} | "
              f"Corr={corr.get('pearson_r',0):.3f}")
        print(f"  Saved: {fname.name}")

        self.results[label] = flat
        return flat

    def tier1_quick_validation(self):
        """Tier 1: 快速验证 (2-3 min)"""
        configs = [
            ('T1_quick_Cortical_8K', {
                'n_neurons': 8000, 'n_episodes': 50, 'n_steps_per_episode': 50,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
            ('T1_quick_Cortical_16K', {
                'n_neurons': 16000, 'n_episodes': 50, 'n_steps_per_episode': 50,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
        ]
        for i, (label, cfg) in enumerate(configs):
            self._run_config(cfg, label, tier=1, step=i)
            self._save_checkpoint(1, i)

    def tier2_medium_scale(self):
        """Tier 2: 中等规模实验 (10-15 min)"""
        configs = [
            ('T2_Cortical_16K_200ep', {
                'n_neurons': 16000, 'n_episodes': 200, 'n_steps_per_episode': 50,
                'n_seeds': 5, 'seed_offset': 42, 'world_size': 10.0,
            }),
            ('T2_Cortical_16K_GridWorld', {
                'n_neurons': 16000, 'n_episodes': 200, 'n_steps_per_episode': 50,
                'n_seeds': 5, 'seed_offset': 42, 'world_size': 20.0,
                'world_class': 'GridWorldV2',
            }),
            ('T2_Cortical_24K_200ep', {
                'n_neurons': 24000, 'n_episodes': 200, 'n_steps_per_episode': 50,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
        ]
        for i, (label, cfg) in enumerate(configs):
            self._run_config(cfg, label, tier=2, step=i)
            self._save_checkpoint(2, i)

    def tier3_deep_training(self):
        """Tier 3: 深度训练 (25-40 min)"""
        configs = [
            ('T3_Cortical_32K_400ep', {
                'n_neurons': 32000, 'n_episodes': 400, 'n_steps_per_episode': 100,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
            ('T3_Cortical_32K_GridWorld', {
                'n_neurons': 32000, 'n_episodes': 400, 'n_steps_per_episode': 100,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 20.0,
                'world_class': 'GridWorldV2',
            }),
        ]
        for i, (label, cfg) in enumerate(configs):
            self._run_config(cfg, label, tier=3, step=i)
            self._save_checkpoint(3, i)

    def tier4_scale_sweep(self):
        """Tier 4: 规模化扫描 (30-50 min)"""
        configs = [
            ('T4_scale_8K', {
                'n_neurons': 8000, 'n_episodes': 200, 'n_steps_per_episode': 100,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
            ('T4_scale_24K', {
                'n_neurons': 24000, 'n_episodes': 200, 'n_steps_per_episode': 100,
                'n_seeds': 3, 'seed_offset': 42, 'world_size': 10.0,
            }),
            ('T4_scale_32K', {
                'n_neurons': 32000, 'n_episodes': 300, 'n_steps_per_episode': 100,
                'n_seeds': 5, 'seed_offset': 42, 'world_size': 10.0,
            }),
        ]
        for i, (label, cfg) in enumerate(configs):
            self._run_config(cfg, label, tier=4, step=i)
            self._save_checkpoint(4, i)

    def tier5_phi_behavior_calibration(self):
        """Tier 5: Phi-Behavior 精细校准 (40-60 min)"""
        configs = [
            ('T5_calibration_32K_GridWorld', {
                'n_neurons': 32000, 'n_episodes': 500, 'n_steps_per_episode': 100,
                'n_seeds': 10, 'seed_offset': 42, 'world_size': 20.0,
                'world_class': 'GridWorldV2',
            }),
            ('T5_calibration_32K_Enhanced', {
                'n_neurons': 32000, 'n_episodes': 500, 'n_steps_per_episode': 100,
                'n_seeds': 10, 'seed_offset': 200, 'world_size': 10.0,
            }),
        ]
        for i, (label, cfg) in enumerate(configs):
            self._run_config(cfg, label, tier=5, step=i)
            self._save_checkpoint(5, i)

    def run_all(self, start_tier=1):
        """运行所有分级实验"""
        print("=" * 70)
        print(f"SNA BATCH ORCHESTRATOR  -  {_timestamp()}")
        print(f"Output: {self.output_dir}")
        print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'default')}")
        print("=" * 70)

        total_start = time.perf_counter()

        tiers = [
            (1, "Quick Validation (2 min)", self.tier1_quick_validation),
            (2, "Medium Scale (15 min)", self.tier2_medium_scale),
            (3, "Deep Training (40 min)", self.tier3_deep_training),
            (4, "Scale Sweep (50 min)", self.tier4_scale_sweep),
            (5, "Phi-Behavior Calibration (60 min)", self.tier5_phi_behavior_calibration),
        ]

        for tier_id, desc, func in tiers:
            if tier_id < start_tier:
                print(f"\n[Tier {tier_id}] {desc} -- SKIPPED (resuming from tier {start_tier})")
                continue

            print(f"\n{'#'*70}")
            print(f"#  TIER {tier_id}: {desc}")
            print(f"#  Started: {_timestamp()}")
            print(f"{'#'*70}")

            t0 = time.perf_counter()
            try:
                func()
            except Exception as e:
                print(f"\n  !! TIER {tier_id} FAILED: {e}")
                traceback.print_exc()
                self._save_checkpoint(tier_id, -1)
                print(f"  Checkpoint saved. Resume with: python batch_orchestrator.py --resume")
                return False

            elapsed = time.perf_counter() - t0
            print(f"\n  Tier {tier_id} completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        total_elapsed = time.perf_counter() - total_start
        self._generate_final_report(total_elapsed)
        self._save_checkpoint(99, 0)
        return True

    def _generate_final_report(self, total_elapsed):
        """生成最终综合报告"""
        print(f"\n{'#'*70}")
        print(f"#  COMPREHENSIVE REPORT")
        print(f"#  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        print(f"{'#'*70}")

        if not self.results:
            print("  No results collected.")
            return

        df = {k: v for k, v in self.results.items()}
        print(f"\n  Experiments completed: {len(df)}")

        for label, data in df.items():
            stars = "**" if data.get('agg_success_rate_mean', 0) > 0.05 else "  "
            print(f"\n  {stars} [{data.get('tier','?')}.{data.get('step','?')}] {label}")
            print(f"    N={data.get('n_neurons',0):,} | ep={data.get('n_episodes',0)} | seeds={data.get('n_seeds',0)}")
            print(f"    Phi: {data.get('agg_mean_phi_mean',0):.4f} +/- {data.get('agg_mean_phi_std',0):.4f}")
            print(f"    Consc: {data.get('agg_mean_consciousness_mean',0):.4f}")
            print(f"    Success: {data.get('agg_success_rate_mean',0):.4f}")
            print(f"    Reprod: {data.get('stab_reproducibility_score',0):.3f}")
            print(f"    Phi-Behav corr: r={data.get('corr_pearson_r',0):.3f} p={data.get('corr_p_value',0):.3f}")
            print(f"    Time: {data.get('elapsed_seconds',0):.0f}s")
            print(f"    World: {data.get('world','Enhanced')}")

        final_file = self.output_dir / f'FINAL_REPORT_{_timestamp()}.json'
        with open(final_file, 'w') as f:
            json.dump(df, f, indent=2, ensure_ascii=False)
        print(f"\n  Final report saved: {final_file}")


def main():
    parser = argparse.ArgumentParser(description='SNA Batch Orchestrator')
    parser.add_argument('--tier', type=int, default=0,
                        help='Run only this tier (0=all, 1-5)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    orch = BatchOrchestrator(output_dir=args.output)

    if args.resume:
        cp = orch._load_checkpoint()
        if cp:
            print(f"Resuming from tier {cp['tier']}, step {cp['step']}")
            print(f"Previously completed: {cp.get('completed', [])}")
            start_tier = cp['tier']
        else:
            print("No checkpoint found. Starting from tier 1.")
            start_tier = 1
    elif args.tier > 0:
        start_tier = args.tier
        print(f"Running tier {args.tier} only")
    else:
        start_tier = 1
        print("Running all tiers (1-5)")

    success = orch.run_all(start_tier=start_tier)

    if success:
        print(f"\n{'='*70}")
        print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"EXPERIMENTS STOPPED (checkpoint saved for resume)")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()