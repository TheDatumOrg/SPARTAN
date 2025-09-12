import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 24})

# --- config (fixed to 'acc') ---
draw_version = 'acc'
root = 'output/classification'
sym_methods = {'SAX': 'sax', 'SFA': 'sfa', 'SPARTAN': 'spartan'}
itrs = ['a4_w2', 'a4_w4', 'a4_w8', 'a4_w12']   # shown on x-axis
ed_method_dir = 'euclidean'
ed_itr = 'euc_1nn'

# x-tick labels
datasets = [r'$\omega=2$', r'$\omega=4$', r'$\omega=8$', r'$\omega=12$']
methods = list(sym_methods.keys())  # ['SAX','SFA','SPARTAN']

# ---- helpers ----
ACC_CANDIDATES = ('acc', 'accuracy', 'test_acc', 'test_accuracy')

def _read_csv_avg_acc(dir_path: str):
    """Read the first CSV in dir_path and return mean of an accuracy-like column."""
    csvs = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
    if not csvs:
        return None
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        cols = [c for c in df.columns if (c.lower() in ACC_CANDIDATES) or ('acc' in c.lower())]
        if not cols:
            continue
        col = cols[0]
        try:
            return float(pd.to_numeric(df[col], errors='coerce').dropna().mean())
        except Exception:
            continue
    return None

def avg_acc_over_ucr(method_fs_name: str, itr_name: str, itr_prefix: str = 'itr_'):
    """
    Walk output/classification/<method_fs_name>/<dataset>/<itr_dir> and average accuracy.
    itr_dir is f'{itr_prefix}{itr_name}' for symbolic methods; for ED pass itr_prefix=''.
    """
    method_root = os.path.join(root, method_fs_name)
    if not os.path.isdir(method_root):
        return float('nan')

    vals = []
    for dataset in sorted(os.listdir(method_root)):
        ds_path = os.path.join(method_root, dataset)
        if not os.path.isdir(ds_path):
            continue
        itr_dir = os.path.join(ds_path, f'{itr_prefix}{itr_name}')
        if not os.path.isdir(itr_dir):
            continue
        v = _read_csv_avg_acc(itr_dir)
        if v is not None and np.isfinite(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else float('nan')

# ---- collect accuracies ----
# accuracy shape: (len(itrs), len(methods))
acc_mat = np.zeros((len(itrs), len(methods)), dtype=float)

for j, (label, fsname) in enumerate(sym_methods.items()):
    for i, itr in enumerate(itrs):
        acc_mat[i, j] = avg_acc_over_ucr(fsname, itr, itr_prefix='itr_')

# Euclidean baseline (single average across datasets)
ED_acc = avg_acc_over_ucr(ed_method_dir, ed_itr, itr_prefix='itr_')

# ---- plotting ----
colors = ['#EAC84E', '#8CC8C5', '#5F7592']
hatches = ['//', '--', '||']

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_axisbelow(True)
ax.grid(color='gray', linestyle='dashed', axis='y')

n_bars = len(methods)
indices = np.arange(len(datasets))
bar_width = 0.15

for i, (method, color, hatch) in enumerate(zip(methods, colors, hatches)):
    bars = ax.bar(indices + i * bar_width, acc_mat[:, i], bar_width, label=method, color=color)
    for bar in bars:
        bar.set_hatch(hatch)

print("symbolic method acc: ", acc_mat)
print("euclidean: ", ED_acc)

# ED horizontal line
if np.isfinite(ED_acc):
    plt.axhline(y=ED_acc, linestyle='--', linewidth=3, label="ED (1-NN)", color='red')

ax.set_ylabel('Accuracy', fontsize=30)
ymin, ymax = 0.4, 0.7  # adjust to your data range

x_sticks = indices + bar_width * (n_bars - 1) / 2
ax.set_xticks(x_sticks)
ax.set_xticklabels(datasets)
ax.set_ylim([ymin, ymax])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.185), ncol=len(methods) + 1)

save_dir = 'result/Section5_8'
os.makedirs(save_dir, exist_ok=True)
save_filename = 'bar_classification_varying_wordlen_acc_ED_line.jpg'
plt.savefig(os.path.join(save_dir, save_filename), dpi=500, bbox_inches='tight')
plt.show()
