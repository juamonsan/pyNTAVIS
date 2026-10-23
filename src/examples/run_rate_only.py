"""
Rate-only check for the Fan NTAS dataset: can balanced and unbalanced loads be told apart from the magnetometer event rate
alone? Uses the same windows and folds as run_all_experiments.py, with three classifiers on the event rates of each window
(single threshold and interval rule on the total rate, k-nearest neighbours on the rate of each axis). Also prints the
recording event-rate ranges per number of magnets and the data rate of the AER stream (file, 16-bit AER bus, 8 used address bits)
compared with the raw sensor stream.

Usage:
    python run_rate_only.py
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
cache, rec_rate = {}, {}


def windows(r, stride):
    key = (r.path, stride)
    if key not in cache:
        f = Loaders.loadAEDAT(r.path, s); Functions.adapt_timestamps(f, s)
        rows = []
        for w in Splitters.window_splitter(f, s, W, stride=stride, min_spikes=MIN_SPIKES):
            a = np.asarray(w.addresses)
            rows.append(np.bincount(np.minimum(a // 86, 2), minlength=3) / (W / 1e6))   # events/s per axis
        cache[key] = np.array(rows)
        rec_rate[r.path] = len(f.addresses) / (f.max_ts / 1e6)
    return cache[key]


def dataset(pairs, stride):
    X, y = [], []
    for r, c in pairs:
        w = windows(r, stride); X.append(w); y += [int(c == 'unbalanced')] * len(w)
    return np.concatenate(X), np.array(y)


def single_threshold(xt, yt, xv):
    best = (0, 0, 1)
    for t in np.unique(np.quantile(xt, np.linspace(0, 1, 201))):
        for sign in (1, -1):
            acc = np.mean((sign * (xt - t) > 0) == yt)
            if acc > best[0]: best = (acc, t, sign)
    _, t, sign = best
    return (sign * (xv - t) > 0).astype(int)


def interval_rule(xt, yt, xv):
    q = np.unique(np.quantile(xt, np.linspace(0, 1, 101)))
    best = (0, q[0], q[-1], 1)
    for i, lo in enumerate(q):
        for hi in q[i + 1:]:
            inside = (xt >= lo) & (xt <= hi)
            for lab in (1, 0):
                acc = np.mean(np.where(inside, lab, 1 - lab) == yt)
                if acc > best[0]: best = (acc, lo, hi, lab)
    _, lo, hi, lab = best
    inside = (xv >= lo) & (xv <= hi)
    return np.where(inside, lab, 1 - lab)


def knn(Xt, yt, Xv, k=15):
    m, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt, Xv = (Xt - m) / sd, (Xv - m) / sd
    out = []
    for i in range(0, len(Xv), 512):
        d = ((Xv[i:i + 512, None, :] - Xt[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out.append((yt[idx].mean(1) > 0.5).astype(int))
    return np.concatenate(out)


print('Magnetometer, unbalance: window accuracy, mean ± std over the 4 folds')
for e in ['2a_unbalance_min', '2a_unbalance_mid', '2a_unbalance_max', '2a_unbalance', '2a_unbalance_all_speeds']:
    acc = {'single threshold (total rate)': [], 'interval rule (total rate)': [], 'kNN (3 per-axis rates)': []}
    for k in range(1, fd.NUM_FOLDS + 1):
        tr, va = fd.select(recs, e, 'Mag', k)
        Xt, yt = dataset(tr, TRAIN_STRIDE); Xv, yv = dataset(va, VAL_STRIDE)
        acc['single threshold (total rate)'].append(np.mean(single_threshold(Xt.sum(1), yt, Xv.sum(1)) == yv))
        acc['interval rule (total rate)'].append(np.mean(interval_rule(Xt.sum(1), yt, Xv.sum(1)) == yv))
        acc['kNN (3 per-axis rates)'].append(np.mean(knn(Xt, yt, Xv) == yv))
    print('  ' + e)
    for name, v in acc.items():
        print('     {:<32} {:5.1f} ± {:4.1f} %   folds {}'.format(name, 100 * np.mean(v), 100 * np.std(v), ' '.join('{:.2f}'.format(x) for x in v)))

print('\nMagnetometer recording event rate (k events/s), no noise, min-max per number of magnets:')
for sp in (fd.MIN, fd.MEDIUM, fd.MAX):
    cells = []
    for m in (3, 4, 5, 6):
        v = [rec_rate[r.path] for r in recs if r.path in rec_rate and r.sensor == 'Mag' and r.speed == sp
             and r.magnets == m and r.external_noise == 'No' and r.internal_noise == 'No']
        cells.append('{}: {:.0f}-{:.0f}'.format(m, min(v) / 1e3, max(v) / 1e3))
    print('  {:<6} '.format(fd.SPEED_NAMES[sp]) + ' | '.join(cells))

# Bus-level data rate: address only, no timestamp
print('\nBus-level data rate (address only), median (range) over all recordings of each sensor:')
raw = {'Accel': 833 * 3 * 16, 'Mag': 560 * 3 * 16}
for sensor in ('Accel', 'Mag'):
    rates = []
    for r in recs:
        if r.sensor != sensor: continue
        if r.path not in rec_rate:
            f = Loaders.loadAEDAT(r.path, s); t = np.asarray(f.timestamps); rec_rate[r.path] = len(t) / ((t.max() - t.min()) / 1e6)
        rates.append(rec_rate[r.path])
    rates = np.array(rates)
    for bits, label in ((64, 'jAER file, 8 bytes/event'), (16, '16-bit AER bus'), (8, '8 used address bits')):
        b = rates * bits
        print('  {:<5} {:<30} {:6.2f} Mbit/s ({:.2f}-{:.2f}) = {:5.1f}x raw ({:.1f}-{:.1f}x)'.format(
            sensor, label, np.median(b) / 1e6, b.min() / 1e6, b.max() / 1e6, np.median(b) / raw[sensor], b.min() / raw[sensor], b.max() / raw[sensor]))
    print('  {:<5} median event rate {:.1f} k events/s; raw stream {:.1f} kbit/s'.format(sensor, np.median(rates) / 1e3, raw[sensor] / 1e3))
