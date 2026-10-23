"""
Normalized-rate check for the Fan NTAS dataset (magnetometer, unbalance): k-nearest neighbours (k = 15) on the event rate
of each axis, with three versions of the features, using the same windows and folds as run_all_experiments.py:
    raw            events/s of each axis in the window;
    fractions      share of each axis in the events of the window (removes the overall rate);
    per-recording  events/s of each axis divided by the mean total rate of the recording.
It includes the cross-speed experiments (2d min <-> max, 2e min+max -> mid) to test whether their results depend on the
change of the event rate with speed. It also prints the mean share of each axis per speed and number of magnets.

Usage:
    python run_rate_normalized.py
"""
import os
import sys
import numpy as np
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_PATH)
import fan_dataset as fd
from pyNAVIS import Functions, Loaders, MainSettings, Splitters

D = os.path.join(SCRIPT_PATH, 'snn_dataset', 'Fan_NTAS_Dataset')
s = MainSettings(num_channels=43, mono_stereo=2, on_off_both=1, address_size=4, ts_tick=1, verbose=False)
W, TRAIN_STRIDE, VAL_STRIDE, MIN_SPIKES = 200_000, 100_000, 200_000, 50
recs = fd.parse_recordings(D, verbose=False)
cache = {}

def windows(r, stride):
    key = (r.path, stride)
    if key not in cache:
        f = Loaders.loadAEDAT(r.path, s); Functions.adapt_timestamps(f, s)
        rows = [np.bincount(np.minimum(np.asarray(w.addresses) // 86, 2), minlength=3) / (W / 1e6)
                for w in Splitters.window_splitter(f, s, W, stride=stride, min_spikes=MIN_SPIKES)]
        rows = np.array(rows)
        cache[key] = (rows, rows.sum(1).mean())          # per-axis rates of each window, mean total rate of the recording
    return cache[key]

def features(rows, rec_mean, kind):
    if kind == 'raw': return rows
    if kind == 'fractions': return rows / rows.sum(1, keepdims=True)
    return rows / rec_mean                               # per-recording normalized

def dataset(pairs, stride, kind):
    X, y = [], []
    for r, c in pairs:
        rows, m = windows(r, stride); X.append(features(rows, m, kind)); y += [int(c == 'unbalanced')] * len(rows)
    return np.concatenate(X), np.array(y)

def knn(Xt, yt, Xv, k=15):
    m, sd = Xt.mean(0), Xt.std(0) + 1e-9; Xt, Xv = (Xt - m) / sd, (Xv - m) / sd
    out = []
    for i in range(0, len(Xv), 512):
        d = ((Xv[i:i + 512, None, :] - Xt[None, :, :]) ** 2).sum(-1)
        out.append((yt[np.argpartition(d, k, axis=1)[:, :k]].mean(1) > 0.5).astype(int))
    return np.concatenate(out)

kinds = ['raw', 'fractions', 'per-recording']
print('Magnetometer, unbalance, kNN (k = 15) on the event rate of each axis: window accuracy, mean ± std over 4 folds')
print('{:<26} {:>18} {:>18} {:>18}'.format('experiment', *kinds))
for e in ['2a_unbalance_min', '2a_unbalance_mid', '2a_unbalance_max', '2a_unbalance_all_speeds',
          '2d_unbalance_min_to_max', '2d_unbalance_max_to_min', '2e_unbalance_unseen_mid']:
    row = []
    for kind in kinds:
        acc = []
        for k in range(1, fd.NUM_FOLDS + 1):
            tr, va = fd.select(recs, e, 'Mag', k)
            Xt, yt = dataset(tr, TRAIN_STRIDE, kind); Xv, yv = dataset(va, VAL_STRIDE, kind)
            acc.append(np.mean(knn(Xt, yt, Xv) == yv))
        row.append('{:5.1f} ± {:4.1f} %'.format(100 * np.mean(acc), 100 * np.std(acc)))
    print('{:<26} {:>18} {:>18} {:>18}'.format(e, *row), flush=True)

print('\nMean axis fractions (X / Y / Z) per number of magnets, magnetometer, no noise:')
for sp in (fd.MIN, fd.MEDIUM, fd.MAX):
    cells = []
    for mg in (3, 4, 5, 6):
        fr = [windows(r, VAL_STRIDE)[0].sum(0) for r in recs if r.sensor == 'Mag' and r.speed == sp and r.magnets == mg
              and r.external_noise == 'No' and r.internal_noise == 'No']
        f = np.sum(fr, axis=0); f = f / f.sum()
        cells.append('{}: {:.2f}/{:.2f}/{:.2f}'.format(mg, *f))
    print('  {:<6} '.format(fd.SPEED_NAMES[sp]) + ' | '.join(cells))
