"""
SNA 自动实验报告生成器 — 汇总所有 experiment_results/*.json 并生成综合报告。

用法:
    python experiments/report_generator.py           # 扫描结果并打印报告
    python experiments/report_generator.py --html    # 生成 HTML 报告
"""
import sys
import os
import json
import glob
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def discover_results(results_dir=None):
    """发现所有实验结果 JSON 文件"""
    if results_dir is None:
        results_dir = ROOT / 'experiment_results'
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    files = list(results_dir.glob('*.json'))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def load_result(filepath):
    """加载单个实验结果, 跳过损坏文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with open(filepath, encoding='utf-8', errors='replace') as f:
            return json.load(f)


def categorize_results(results):
    """按 tier 分类所有结果"""
    by_tier = defaultdict(list)
    phi_failures = []
    all_success_rates = []

    for filepath, data in results:
        tier = data.get('tier', 99)
        by_tier[tier].append((filepath, data))

        agg = {k: v for k, v in data.items() if k.startswith('agg_')}
        corr = {k: v for k, v in data.items() if k.startswith('corr_')}

        if agg:
            sr = agg.get('agg_success_rate_mean', 0)
            if sr is not None:
                all_success_rates.append(sr)

        r = corr.get('corr_pearson_r', 0)
        if r is not None and abs(r) < 0.2 and data.get('n_seeds', 0) >= 3:
            phi_failures.append((filepath.name, r))

    by_tier = dict(sorted(by_tier.items()))
    return by_tier, phi_failures, all_success_rates


def compute_tier_summary(tier_id, items):
    """计算单个 tier 的汇总统计"""
    phis = []
    success_rates = []
    concs = []
    reprods = []
    correlations = []

    for _, data in items:
        phi = data.get('agg_mean_phi_mean')
        sr = data.get('agg_success_rate_mean')
        conc = data.get('agg_mean_consciousness_mean')
        reprod = data.get('stab_reproducibility_score')
        corr = data.get('corr_pearson_r')

        if phi is not None: phis.append(phi)
        if sr is not None: success_rates.append(sr)
        if conc is not None: concs.append(conc)
        if reprod is not None: reprods.append(reprod)
        if corr is not None: correlations.append(corr)

    def stats(arr):
        if not arr: return (0, 0, 0, 0)
        a = np.array(arr)
        return (float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0),
                float(np.min(a)), float(np.max(a)))

    return {
        'n_experiments': len(items),
        'phi': stats(phis),
        'success_rate': stats(success_rates),
        'consciousness': stats(concs),
        'reproducibility': stats(reprods),
        'phi_behavior_corr': stats(correlations),
    }


def generate_text_report(results_dir=None):
    """生成文本报告"""
    files = discover_results(results_dir)
    if not files:
        print("No experiment results found.")
        return ""

    results = [(f, load_result(f)) for f in files]
    by_tier, phi_failures, all_sr = categorize_results(results)

    lines = []
    lines.append("=" * 70)
    lines.append(f"SNA COMPREHENSIVE EXPERIMENT REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Results directory: {results_dir or ROOT / 'experiment_results'}")
    lines.append(f"Total result files: {len(files)}")
    lines.append("=" * 70)

    lines.append(f"\n  {'─' * 55}")

    if all_sr:
        sr_arr = np.array(all_sr)
        lines.append(f"\n  OVERALL STATISTICS")
        lines.append(f"    Total experiments: {len(files)}")
        lines.append(f"    Max success rate:  {np.max(sr_arr):.4f}")
        lines.append(f"    Mean success rate: {np.mean(sr_arr):.4f}")
        lines.append(f"    Min success rate:  {np.min(sr_arr):.4f}")
        high_perf = sum(1 for s in all_sr if s > 0.1)
        lines.append(f"    High-performance (>10%): {high_perf}/{len(all_sr)}")

    for tier_id in sorted(by_tier.keys()):
        items = by_tier[tier_id]
        summary = compute_tier_summary(tier_id, items)
        n = summary['n_experiments']
        phi_m, phi_s, _, _ = summary['phi']
        sr_m, sr_s, _, _ = summary['success_rate']
        conc_m, conc_s, _, _ = summary['consciousness']
        reprod_m, _, _, _ = summary['reproducibility']
        corr_m, _, _, _ = summary['phi_behavior_corr']

        tier_names = {
            1: 'Quick Validation', 2: 'Medium Scale', 3: 'Deep Training',
            4: 'Scale Sweep', 5: 'Phi-Behavior Calibration', 99: 'Standalone'
        }
        lines.append(f"\n  {'─' * 55}")
        lines.append(f"\n  TIER {tier_id}: {tier_names.get(tier_id, 'Unknown')} ({n} experiments)")
        lines.append(f"    Phi:                 {phi_m:.4f} +/- {phi_s:.4f}")
        lines.append(f"    Success Rate:        {sr_m:.4f} +/- {sr_s:.4f}")
        lines.append(f"    Consciousness:       {conc_m:.4f} +/- {conc_s:.4f}")
        lines.append(f"    Reproducibility:     {reprod_m:.4f}")
        lines.append(f"    Phi-Behavior Corr:   {corr_m:.4f}")

        for filepath, data in items:
            label = data.get('label', filepath.name)
            elapsed = data.get('elapsed_seconds', 0)
            n_neurons = data.get('n_neurons', 0)
            phi_v = data.get('agg_mean_phi_mean', 0)
            sr_v = data.get('agg_success_rate_mean', 0)
            corr_v = data.get('corr_pearson_r', 0)
            reprod_v = data.get('stab_reproducibility_score', 0)
            star = " *" if sr_v > 0.05 else ""
            lines.append(f"      [{n_neurons//1000}K, {elapsed:.0f}s]{star} {label}: phi={phi_v:.4f} sr={sr_v:.4f} corr={corr_v:.3f} repr={reprod_v:.3f}")

    if phi_failures:
        lines.append(f"\n  {'─' * 55}")
        lines.append(f"\n  PHI-BEHAVIOR WEAK CORRELATION ({len(phi_failures)}):")
        for name, r in phi_failures[:5]:
            lines.append(f"    r={r:.3f}  {name}")

    lines.append(f"\n{'=' * 70}")
    lines.append(f"END OF REPORT")
    lines.append(f"{'=' * 70}")

    text = '\n'.join(lines)
    return text


def generate_html_report(results_dir=None):
    """生成 HTML 报告"""
    files = discover_results(results_dir)
    if not files:
        return "<html><body><h1>No experiment results found.</h1></body></html>"

    results = [(f, load_result(f)) for f in files]
    by_tier, phi_failures, all_sr = categorize_results(results)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>SNA Experiment Report</title>
<style>
body {{ font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #7ee787; border-bottom: 1px solid #30363d; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #30363d; padding: 6px 12px; text-align: right; }}
th {{ background: #161b22; color: #8b949e; }}
tr:hover {{ background: #1c2128; }}
.good {{ color: #7ee787; }}
.bad {{ color: #f85149; }}
.warn {{ color: #d2991d; }}
.summary {{ background: #161b22; padding: 15px; border-radius: 8px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>SNA Comprehensive Experiment Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Files: {len(files)}</p>
"""

    if all_sr:
        sr_arr = np.array(all_sr)
        html += f"""
<div class="summary">
<h2>Overall Statistics</h2>
<p>Max success rate: <span class="good">{np.max(sr_arr):.4f}</span> | 
Mean: <span>{np.mean(sr_arr):.4f}</span> | Min: <span class="bad">{np.min(sr_arr):.4f}</span></p>
</div>
"""

    for tier_id in sorted(by_tier.keys()):
        items = by_tier[tier_id]
        summary = compute_tier_summary(tier_id, items)
        tier_names = {1: 'Quick Validation', 2: 'Medium Scale', 3: 'Deep Training',
                       4: 'Scale Sweep', 5: 'Phi-Behavior Calibration', 99: 'Standalone'}

        html += f"<h2>Tier {tier_id}: {tier_names.get(tier_id, 'Unknown')}</h2>"
        html += f"<table><tr><th>Label</th><th>N</th><th>Episodes</th><th>Seeds</th><th>Phi</th><th>Success</th><th>Consc</th><th>Reprod</th><th>Corr(r)</th><th>Time(s)</th></tr>"

        for _, data in items:
            label = data.get('label', '?')[:40]
            n = data.get('n_neurons', 0) // 1000
            ep = data.get('n_episodes', 0)
            seeds = data.get('n_seeds', 0)
            phi = data.get('agg_mean_phi_mean', 0) or 0
            sr = data.get('agg_success_rate_mean', 0) or 0
            conc = data.get('agg_mean_consciousness_mean', 0) or 0
            reprod = data.get('stab_reproducibility_score', 0) or 0
            corr = data.get('corr_pearson_r', 0) or 0
            elapsed = data.get('elapsed_seconds', 0) or 0
            sr_cls = 'good' if sr > 0.05 else ('warn' if sr > 0.01 else 'bad')
            html += f"<tr><td>{label}</td><td>{n}K</td><td>{ep}</td><td>{seeds}</td><td>{phi:.4f}</td><td class='{sr_cls}'>{sr:.4f}</td><td>{conc:.4f}</td><td>{reprod:.3f}</td><td>{corr:.3f}</td><td>{elapsed:.0f}</td></tr>"

        html += "</table>"

    html += "</body></html>"
    return html


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SNA Experiment Report Generator')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory to scan')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()

    if args.html:
        html_content = generate_html_report(args.results_dir)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved: {args.output}")
        else:
            output_path = Path(args.results_dir or (ROOT / 'experiment_results'))
            output_path.mkdir(parents=True, exist_ok=True)
            fname = output_path / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved: {fname}")
    else:
        text = generate_text_report(args.results_dir)
        print(text)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\nText report saved: {args.output}")


if __name__ == '__main__':
    main()