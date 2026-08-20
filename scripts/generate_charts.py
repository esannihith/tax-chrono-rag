import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path('docs/assets')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. RETRIEVAL BENCHMARK CHART
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

k_vals = [5, 10, 20]
hit_rate_hybrid = [83.33, 100.00, 100.00]
hit_rate_random = [5.15, 10.17, 19.81]

recall_hybrid = [80.56, 99.31, 100.00]
recall_random = [2.73, 5.46, 10.93]

ndcg_hybrid = [0.8720, 0.9003, 0.9059]

x = np.arange(len(k_vals))
width = 0.35

ax1.bar(x - width/2, hit_rate_hybrid, width, label='Legal Hybrid (Parent-Aware)', color='#1f77b4')
ax1.bar(x + width/2, hit_rate_random, width, label='Random Baseline (N=183)', color='#aec7e8')
ax1.set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Top-k Hit Rate vs. Random Baseline', fontsize=14, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels([f'Top-{k}' for k in k_vals], fontsize=11)
ax1.set_ylim(0, 115)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
for i, v in enumerate(hit_rate_hybrid):
    ax1.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
for i, v in enumerate(hit_rate_random):
    ax1.text(i + width/2, v + 2, f'{v:.1f}%', ha='center', fontsize=9)

ax2.plot(k_vals, recall_hybrid, marker='o', linewidth=2.5, color='#2ca02c', label='Essential Recall@k')
ax2.plot(k_vals, [v * 100 for v in ndcg_hybrid], marker='s', linewidth=2.5, color='#ff7f0e', label='Graded NDCG@k (x100)')
ax2.plot(k_vals, recall_random, marker='^', linewidth=1.5, linestyle='--', color='#d62728', label='Random Recall Baseline')
ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Retrieval Quality Trajectory (Active Suite N=90)', fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(k_vals)
ax2.set_xticklabels([f'k={k}' for k in k_vals], fontsize=11)
ax2.set_ylim(0, 115)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
for k, r in zip(k_vals, recall_hybrid):
    ax2.text(k, r + 2.5, f'{r:.1f}%', ha='center', fontweight='bold', color='#2ca02c', fontsize=10)

plt.tight_layout()
fig.savefig(out_dir / 'retrieval_benchmark_curve.png', bbox_inches='tight')
plt.close(fig)
print('Generated docs/assets/retrieval_benchmark_curve.png')

# 2. GENERATION EVALUATION BREAKDOWN CHART
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

metrics = [
    'Negative Abstention\nAccuracy',
    'Criteria Keyword\nCoverage',
    'Citation\nRecall',
    'Composite\nScore',
    'Citation\nPrecision',
    'Strict Temporal\nAccuracy (Labelled)'
]
scores = [100.0, 67.8, 55.0, 54.0, 50.0, 25.0]
colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', '#8c564b', '#17becf']

bars = ax.barh(metrics, scores, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
ax.set_xlim(0, 115)
ax.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Multi-Dimensional Statutory Generation Benchmark (N=20 Stratified)', fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
fig.savefig(out_dir / 'generation_evaluation_breakdown.png', bbox_inches='tight')
plt.close(fig)
print('Generated docs/assets/generation_evaluation_breakdown.png')

