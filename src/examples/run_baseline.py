"""
Non-spiking baseline for the Fan NTAS experiments: a logistic regression on simple features of each window, with the same
windows and folds as run_all_experiments.py. It checks whether a task is hard because the information is missing from the
data, or because of the SNN configuration.

Features of each 200 ms window (computed from the same 1 ms frames as the SNN input):
    - events per address, normalized by the total events of the window (which addresses fire, 258 values);
    - event-rate spectrum of each axis from 5 to 100 Hz, relative to the mean rate (the rhythm, 3 x 20 values).

Usage:
    python run_baseline.py                                    # default selection of experiments, both sensors
    python run_baseline.py --experiments 2a_unbalance_min --sensors Accel

Prints the mean ± standard deviation of the window accuracy over the folds and saves baseline_results.json in the output folder.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from pyNAVIS import Functions, Loaders, MainSettings, Splitters

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_PATH)
from fan_dataset import EXPERIMENTS, NUM_FOLDS, SENSORS, fold_report, parse_recordings, select  # noqa: E402

# Same windows as the SNN (run_all_experiments.py)
settings = MainSettings(num_channels=43, mono_stereo=MainSettings.THREE_AXIS, on_off_both=1, address_size=4, ts_tick=1,
                        verbose=False)
WINDOW_SIZE, TRAIN_STRIDE, VALIDATION_STRIDE, TIME_STEP, MIN_SPIKES = 200_000, 100_000, 200_000, 1_000, 50
BLOCK = settings.addresses_per_source

DEFAULT_EXPERIMENTS = ['2a_unbalance_min', '2a_unbalance_mid', '2d_unbalance_min_to_max', '3_load_min', '3_load_mid',
                       '3_load', '5_noise', '6d_tape_mid']

_cache = {}


def window_features(path, stride):
    """Features of every window of a recording (cached: a recording appears in several folds)."""
    key = (path, stride)
    if key not in _cache:
        recording = Loaders.loadAEDAT(path, settings)
        Functions.adapt_timestamps(recording, settings)
        features = []
        for window in Splitters.window_splitter(recording, settings, WINDOW_SIZE, stride=stride, min_spikes=MIN_SPIKES):
            frames = Functions.spikes_to_frames(window, settings, TIME_STEP, duration=WINDOW_SIZE, start_ts=0).astype(np.float64)
            counts = frames.sum(axis=0)
            rates = frames.reshape(frames.shape[0], settings.num_sources, BLOCK).sum(axis=2)        # [steps, axes]
            spectrum = np.abs(np.fft.rfft(rates - rates.mean(axis=0), axis=0))[1:21]             # 5-100 Hz
            spectrum /= rates.mean(axis=0) * frames.shape[0] + 1e-9
            features.append(np.concatenate([counts / max(counts.sum(), 1), spectrum.T.ravel()]))
        _cache[key] = np.array(features, dtype=np.float32)
    return _cache[key]


def build_set(pairs, classes, stride):
    X, y = [], []
    for recording, class_name in pairs:
        f = window_features(recording.path, stride)
        X.append(f); y += [classes.index(class_name)] * len(f)
    return np.concatenate(X), np.array(y)


def logistic_regression(X_train, y_train, X_val, num_classes, epochs=300):
    torch.manual_seed(0)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    Xt = torch.tensor((X_train - mean) / std); Xv = torch.tensor((X_val - mean) / std); yt = torch.tensor(y_train)
    model = torch.nn.Linear(Xt.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(model(Xt), yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return model(Xv).argmax(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser(description='Non-spiking logistic-regression baseline for the Fan NTAS experiments.')
    parser.add_argument('--experiments', nargs='+', default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--sensors', nargs='+', default=SENSORS, choices=SENSORS)
    parser.add_argument('--dataset', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'Fan_NTAS_Dataset'))
    parser.add_argument('--output', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'experiments', 'script_output'))
    args = parser.parse_args()

    recordings = parse_recordings(args.dataset, verbose=False)
    results = {}
    for experiment in args.experiments:
        for sensor in args.sensors:
            accuracies, chance = [], None
            for fold in range(1, NUM_FOLDS + 1):
                if not fold_report(recordings, experiment, sensor)[fold]['complete']:
                    continue
                train, validation = select(recordings, experiment, sensor, fold)
                classes = sorted({c for _, c in train})
                X_train, y_train = build_set(train, classes, TRAIN_STRIDE)
                X_val, y_val = build_set(validation, classes, VALIDATION_STRIDE)
                predictions = logistic_regression(X_train, y_train, X_val, len(classes))
                accuracies.append(float(np.mean(predictions == y_val)))
                chance = 1 / len(classes)
            results['{}/{}'.format(experiment, sensor)] = dict(fold_accuracy=accuracies, mean=float(np.mean(accuracies)),
                                                              std=float(np.std(accuracies)), chance=chance)
            print('{:<26} {:<6} baseline window accuracy {:5.1f} ± {:4.1f} %  (chance {:.0f} %, folds {})'.format(
                experiment, sensor, 100 * np.mean(accuracies), 100 * np.std(accuracies), 100 * chance,
                ' '.join('{:.2f}'.format(a) for a in accuracies)), flush=True)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved', os.path.join(args.output, 'baseline_results.json'))


if __name__ == '__main__':
    main()
