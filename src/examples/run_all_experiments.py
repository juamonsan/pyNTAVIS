"""
Runs the cross-validation of section 11 of pyNTAVIS_SNN_training_snnTorch.ipynb for all the Fan NTAS experiments at once,
and exports the results and figures of every experiment, sensor and fold.

The parameters, network, training and evaluation are the same as in the notebook (sections 3 and 5-10). If you change them
in the notebook, change them here too (see "Parameters" below).

Usage (from any folder):
    python run_all_experiments.py                                   # all experiments, both sensors, all folds
    python run_all_experiments.py --experiments 2a_unbalance_min 3_load_min --sensors Mag
    python run_all_experiments.py --skip-existing                   # resume an interrupted run
    python run_all_experiments.py --list                            # list the available experiments
    python run_all_experiments.py --check-folds                     # show which folds can be run with the current data

Output (default: snn_dataset/experiments/script_output/, next to this script):
    summary.json, summary.csv          one row per finished experiment and sensor (updated after each of them)
    run_<date>_<time>.log              everything printed during a run
    <experiment>/
        snn_architecture.svg, .png     figure of the network (section 7 of the notebook)
        <sensor>/
            cv_results.json            mean ± standard deviation and the results of every fold
            fold<k>/
                results.json           settings and accuracy of the fold (as in the notebook)
                training_curves.svg, .png
                confusion_matrix.svg, .png
                snn_model.pt           best model of the fold
                data/                  recordings and windows of the fold (deleted at the end unless --keep-data)

If training fails in a fold (stuck at the chance level in both attempts), the fold is reported as failed, left out of
the averages, and its training curves are saved as training_curves_failed.svg / .png to help reviewing the data.
Folds without training or validation recordings in the current dataset (for example, combinations of medium speed that
don't have NUM_FOLDS repetitions yet) are reported as skipped and also left out of the averages.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')   # no windows: figures are only saved to files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate

from pyNAVIS import Functions, Loaders, MainSettings, Splitters

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_PATH)
from fan_dataset import (EXPERIMENTS, NUM_FOLDS, SENSORS, EmptyFoldError, build_experiment, fold_report,  # noqa: E402
                         parse_recordings, select)


# ---------------------------------------------------------------------------------------------------------------------
# Parameters (same values as in the notebook)
# ---------------------------------------------------------------------------------------------------------------------

# Section 3: windows and frames
settings = MainSettings(num_channels=43, mono_stereo=MainSettings.THREE_AXIS, on_off_both=1,
                        address_size=4, ts_tick=1, bin_size=20000, verbose=False)
WINDOW_SIZE = 200_000        # 200 ms
TRAIN_STRIDE = 100_000       # 50 % overlap
VALIDATION_STRIDE = WINDOW_SIZE
TIME_STEP = 1_000            # 1 ms -> 200 time steps per window
MIN_SPIKES = 50
BINARY_FRAMES = False
NUM_ADDRESSES = settings.addresses_per_source * settings.num_sources
NUM_STEPS = int(np.ceil(WINDOW_SIZE / TIME_STEP))

# Section 6: data loading
BATCH_SIZE = 16

# Section 7: network
NUM_HIDDEN = 128
BETA = 0.9              # membrane decay rate per time step
CORRECT_RATE = 0.8      # target firing rate of the output neuron of the correct class (fraction of time steps)
INCORRECT_RATE = 0.2    # target firing rate of the other output neurons

# Section 8: training
NUM_EPOCHS = 20
LEARNING_RATE = 2e-3
RANDOM_SEED = 0          # seed of the first attempt; a restart uses RANDOM_SEED + 1
RESTART_PATIENCE = 5     # epochs to wait before deciding that training is stuck
RESTART_MARGIN = 0.05    # training accuracy must rise this much above the chance level
MAX_ATTEMPTS = 2         # first attempt + one restart

# Section 11: cross-validation
CROSS_SESSION_FOLD = 4   # validates on 14 October, a session that is not used for training

FIGURE_FORMATS = ['svg', 'png']
PNG_DPI = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = SF.mse_count_loss(correct_rate=CORRECT_RATE, incorrect_rate=INCORRECT_RATE)

_log_file = None


def log(message=''):
    """Prints a message and appends it to the log file of the run."""
    print(message, flush=True)
    if _log_file is not None:
        with open(_log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


def remove_folder(path):
    """Deletes a folder. Retries once (network and NTFS drives can be slow to release files), then only warns."""
    for attempt in range(2):
        try:
            shutil.rmtree(path)
            return
        except OSError as error:
            if attempt == 0:
                time.sleep(2)
            else:
                log('    Warning: could not delete {} ({}). Delete it manually.'.format(path, error))


def save_figure(fig, path_without_extension):
    for extension in FIGURE_FORMATS:
        fig.savefig('{}.{}'.format(path_without_extension, extension), dpi=PNG_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Data preparation (section 5 of the notebook)
# ---------------------------------------------------------------------------------------------------------------------

def prepare_dataset(source_path, output_path):
    """Windows every recording of source_path/train and source_path/validation and saves train.npz, validation.npz and config.json in output_path."""
    class_names = sorted(d for d in os.listdir(os.path.join(source_path, 'train'))
                         if os.path.isdir(os.path.join(source_path, 'train', d)))

    def prepare_split(split, stride):
        split_path = os.path.join(source_path, split)
        unknown = set(os.listdir(split_path)) - set(class_names)
        if unknown:
            log('    Warning: classes not present in train are ignored: {}'.format(sorted(unknown)))

        frames, labels, recordings, recording_names = [], [], [], []
        for label, class_name in enumerate(class_names):
            for path in sorted(glob.glob(os.path.join(split_path, class_name, '*.aedat'))):
                spikes_file = Loaders.loadAEDAT(path, settings)
                Functions.adapt_timestamps(spikes_file, settings)
                windows = Splitters.window_splitter(spikes_file, settings, WINDOW_SIZE, stride=stride, min_spikes=MIN_SPIKES)
                for window in windows:
                    frames.append(Functions.spikes_to_frames(window, settings, TIME_STEP, duration=WINDOW_SIZE,
                                                             start_ts=0, binary=BINARY_FRAMES))
                    labels.append(label)
                    recordings.append(len(recording_names))
                recording_names.append(os.path.relpath(path, source_path))

        data = dict(frames=np.stack(frames), labels=np.array(labels), recordings=np.array(recordings),
                    recording_names=np.array(recording_names), class_names=np.array(class_names))
        np.savez_compressed(os.path.join(output_path, split + '.npz'), **data)
        return data

    os.makedirs(output_path, exist_ok=True)
    train_data = prepare_split('train', TRAIN_STRIDE)
    validation_data = prepare_split('validation', VALIDATION_STRIDE)

    config = dict(num_channels=settings.num_channels, mono_stereo=settings.mono_stereo, on_off_both=settings.on_off_both,
                  address_size=settings.address_size, ts_tick=settings.ts_tick, window_size_us=WINDOW_SIZE,
                  train_stride_us=TRAIN_STRIDE, validation_stride_us=VALIDATION_STRIDE, time_step_us=TIME_STEP,
                  min_spikes=MIN_SPIKES, binary_frames=BINARY_FRAMES, class_names=class_names, source_path=source_path)
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return train_data, validation_data


def make_loaders(output_path):
    """Loads train.npz and validation.npz from output_path and wraps them in DataLoaders (section 6)."""
    loaded = {}
    for split in ['train', 'validation']:
        with np.load(os.path.join(output_path, split + '.npz')) as npz:     # read everything and close the file
            data = {key: npz[key] for key in npz.files}
        dataset = TensorDataset(torch.tensor(data['frames'], dtype=torch.float32),
                                torch.tensor(data['labels'], dtype=torch.long))
        loaded[split] = (data, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train')))
    return loaded['train'][0], loaded['validation'][0], loaded['train'][1], loaded['validation'][1]


# ---------------------------------------------------------------------------------------------------------------------
# Network (section 7 of the notebook)
# ---------------------------------------------------------------------------------------------------------------------

class SpikingNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x: [batch, time steps, inputs] -> output spikes: [time steps, batch, outputs]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_out = []
        for step in range(x.shape[1]):
            spk1, mem1 = self.lif1(self.fc1(x[:, step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk_out.append(spk2)
        return torch.stack(spk_out)


def plot_snn_architecture(net, class_names, max_drawn=7):
    """Draws the layers of a SpikingNet: input addresses (X, Y, Z blocks), hidden LIF layer and output LIF layer."""
    num_inputs, num_hidden, num_outputs = net.fc1.in_features, net.fc1.out_features, net.fc2.out_features
    beta = float(net.lif1.beta)
    block = settings.addresses_per_source
    axis_colors = ['tab:red', 'tab:green', 'tab:blue']

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(-0.6, 3.95)
    ax.set_ylim(-1.35, 1.25)
    ax.axis('off')

    def column(x, n, drawn, top=0.9, bottom=-0.9):
        """Vertical positions of the drawn neurons of a layer, with a gap for the ones that are not drawn."""
        if n <= drawn:
            return list(np.linspace(top, bottom, n) if n > 1 else [0.0]), None
        ys = list(np.linspace(top, bottom, drawn + 1))
        gap = ys.pop(drawn // 2)
        return ys, gap

    # Input layer: a few addresses of each axis block
    input_ys, input_colors = [], []
    per_axis = 3
    for axis, (label, color) in enumerate(zip(settings.source_labels, axis_colors)):
        top = 0.95 - axis * 0.66
        ys = np.linspace(top, top - 0.42, per_axis)
        input_ys += list(ys)
        input_colors += [color] * per_axis
        ax.text(-0.18, ys.mean(), '{}\n{} addresses'.format(label, block), ha='right', va='center', fontsize=9, color=color)
        ax.text(0, ys[-1] - 0.1, '⋮', ha='center', va='center', fontsize=12, color=color)

    hidden_ys, hidden_gap = column(1.5, num_hidden, max_drawn)
    output_ys, _ = column(3.0, num_outputs, max_drawn, top=0.25 * (num_outputs - 1), bottom=-0.25 * (num_outputs - 1))

    # Connections (fully connected layers)
    for y0 in input_ys:
        for y1 in hidden_ys:
            ax.plot([0, 1.5], [y0, y1], color='0.8', linewidth=0.4, zorder=1)
    for y0 in hidden_ys:
        for y1 in output_ys:
            ax.plot([1.5, 3.0], [y0, y1], color='0.6', linewidth=0.6, zorder=1)

    # Neurons
    ax.scatter([0] * len(input_ys), input_ys, s=140, c=input_colors, edgecolors='black', linewidths=0.6, zorder=3)
    ax.scatter([1.5] * len(hidden_ys), hidden_ys, s=260, c='gold', edgecolors='black', linewidths=0.6, zorder=3)
    if hidden_gap is not None:
        ax.text(1.5, hidden_gap, '⋮', ha='center', va='center', fontsize=16)
    ax.scatter([3.0] * len(output_ys), output_ys, s=420, c='orchid', edgecolors='black', linewidths=0.6, zorder=3)
    for y, name in zip(output_ys, class_names):
        ax.text(3.17, y, name, ha='left', va='center', fontsize=10)
    ax.text(3.0, -1.02, 'Training target (mse_count_loss):\ncorrect class fires in {:.0%} of the steps,\nthe other classes in {:.0%}'.format(
                CORRECT_RATE, INCORRECT_RATE),
            ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='lavenderblush', edgecolor='0.6'))

    # Layer titles and connection labels
    ax.text(0, 1.18, 'Input\n{} addresses'.format(num_inputs), ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 1.18, 'Hidden layer\n{} LIF neurons'.format(num_hidden), ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.0, 1.18, 'Output layer\n{} LIF neurons'.format(num_outputs), ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(0.75, 1.18, 'fc1: fully connected\n{} × {} weights'.format(num_inputs, num_hidden), ha='center', va='center', fontsize=9)
    ax.text(2.25, 1.18, 'fc2: fully connected\n{} × {} weights'.format(num_hidden, num_outputs), ha='center', va='center', fontsize=9)
    ax.text(1.5, -1.02, 'LIF neuron:  $U[t] = \\beta\\, U[t-1] + I[t]$\nspike and reset when $U[t] > 1$', ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='0.6'))

    ax.text(1.5, -1.3, 'Simulated for {} time steps of {:g} ms (one row of the window per step). LIF decay β = {:.2f}. '
            'Predicted class = output neuron with the most spikes.'.format(NUM_STEPS, TIME_STEP / 1000, beta),
            ha='center', va='center', fontsize=9, style='italic')
    ax.set_title('SNN architecture', fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# Training (section 8 of the notebook)
# ---------------------------------------------------------------------------------------------------------------------

class TrainingFailedError(RuntimeError):
    """Raised when every training attempt stays at the chance level. Its history attribute has the last attempt."""

    def __init__(self, message, history=None):
        super().__init__(message)
        self.history = history


def run_epoch(net, optimizer, loader, train):
    net.train(train)
    total_loss, total_correct, total = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            spk_out = net(frames)
            loss = loss_fn(spk_out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += SF.accuracy_rate(spk_out, labels) * len(labels)
            total += len(labels)
    return total_loss / total, total_correct / total


def train_model(train_loader, validation_loader, num_classes, output_path):
    """
    Trains a new SpikingNet and returns the model with the lowest validation loss, its history, its epoch and the seed used.

    If the training accuracy is still at the chance level after RESTART_PATIENCE epochs, training restarts once with
    another seed. If every attempt is stuck, it raises TrainingFailedError.
    """
    model_path = os.path.join(output_path, 'snn_model.pt')
    train_labels = train_loader.dataset.tensors[1]
    chance_accuracy = torch.bincount(train_labels, minlength=num_classes).max().item() / len(train_labels)  # most frequent class

    for attempt in range(MAX_ATTEMPTS):
        seed = RANDOM_SEED + attempt
        torch.manual_seed(seed)
        net = SpikingNet(NUM_ADDRESSES, NUM_HIDDEN, num_classes, BETA).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss, best_epoch, stuck = float('inf'), None, False
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = run_epoch(net, optimizer, train_loader, train=True)
            val_loss, val_acc = run_epoch(net, optimizer, validation_loader, train=False)
            for key, value in zip(history, [train_loss, train_acc, val_loss, val_acc]):
                history[key].append(value)

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch + 1
                torch.save(net.state_dict(), model_path)

            # Stuck: the training accuracy never rose clearly above the chance level
            if epoch + 1 == RESTART_PATIENCE and max(history['train_acc']) < chance_accuracy + RESTART_MARGIN:
                stuck = True
                break

        if not stuck:
            break
        message = ('Training is stuck with seed {}: after {} epochs the training accuracy ({:.3f}) is still at the chance '
                   'level ({:.3f}), so the network is not learning.').format(
                       seed, RESTART_PATIENCE, max(history['train_acc']), chance_accuracy)
        if attempt + 1 < MAX_ATTEMPTS:
            log('    WARNING: {} Restarting with seed {}.'.format(message, seed + 1))
        else:
            raise TrainingFailedError(
                '{}\nTraining stopped after {} attempts (seeds {} to {}). Please review the data:\n'
                '  - check that the classes really differ in these recordings and with this sensor\n'
                '    (for example, the accelerometer at min speed does not capture the unbalance);\n'
                '  - check the labels (folder names) and the MainSettings of the recordings;\n'
                '  - check that the windows contain spikes (WINDOW_SIZE, MIN_SPIKES, TIME_STEP);\n'
                '  - if the data is correct, try a different LEARNING_RATE, BETA, NUM_HIDDEN or CORRECT_RATE / INCORRECT_RATE.'.format(
                    message, MAX_ATTEMPTS, RANDOM_SEED, RANDOM_SEED + MAX_ATTEMPTS - 1), history)

    net.load_state_dict(torch.load(model_path, map_location=device))
    return net, history, best_epoch, seed


def plot_training_curves(history, title):
    """Loss and accuracy per epoch (section 9 of the notebook)."""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history['train_loss'], label='Train')
    ax_loss.plot(epochs, history['val_loss'], label='Validation')
    ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss'); ax_loss.set_title('Loss'); ax_loss.legend()
    ax_acc.plot(epochs, history['train_acc'], label='Train')
    ax_acc.plot(epochs, history['val_acc'], label='Validation')
    ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy'); ax_acc.set_title('Accuracy (per window)'); ax_acc.legend()
    ax_acc.set_ylim(0, 1.05)
    for ax in (ax_loss, ax_acc):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation (section 10 of the notebook)
# ---------------------------------------------------------------------------------------------------------------------

def evaluate(net, validation_loader, validation_npz, output_path, best_epoch, seed):
    """Classifies every validation window and every recording (majority vote) and saves results.json in output_path."""
    class_names = list(validation_npz['class_names'])
    net.eval()
    with torch.no_grad():
        window_predictions = torch.cat([net(frames.to(device)).sum(dim=0).argmax(dim=1).cpu()
                                        for frames, _ in validation_loader]).numpy()

    labels = validation_npz['labels']
    recordings = validation_npz['recordings']
    window_accuracy = float(np.mean(window_predictions == labels))

    correct_recordings, recording_predictions = 0, []
    for i, name in enumerate(validation_npz['recording_names']):
        mask = recordings == i
        if not mask.any():
            continue
        votes = np.bincount(window_predictions[mask], minlength=len(class_names))
        predicted, true = votes.argmax(), labels[mask][0]
        correct_recordings += predicted == true
        recording_predictions.append(dict(recording=str(name), true=class_names[true], predicted=class_names[predicted],
                                          votes=int(votes[predicted]), windows=int(mask.sum())))

    num_recordings = len(np.unique(recordings))
    recording_accuracy = float(correct_recordings / num_recordings)

    with open(os.path.join(output_path, 'config.json')) as f:
        results = json.load(f)
    results.update(num_hidden=NUM_HIDDEN, beta=BETA, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
                   batch_size=BATCH_SIZE, loss='mse_count_loss', correct_rate=CORRECT_RATE, incorrect_rate=INCORRECT_RATE,
                   random_seed=seed, best_epoch=best_epoch, window_accuracy=window_accuracy,
                   recording_accuracy=recording_accuracy, num_validation_windows=int(len(labels)),
                   num_validation_recordings=int(num_recordings), recording_predictions=recording_predictions)
    return results, window_predictions


def plot_confusion_matrix(labels, window_predictions, class_names, title):
    """Confusion matrix of the validation windows, normalised per true class (section 10 of the notebook)."""
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(confusion, (labels, window_predictions), 1)
    normalised = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(1.2 * num_classes + 3, 1.0 * num_classes + 2))
    plt.imshow(normalised, cmap='Blues', vmin=0, vmax=1)
    for true in range(num_classes):
        for predicted in range(num_classes):
            plt.text(predicted, true, '{:.2f}\n({})'.format(normalised[true, predicted], confusion[true, predicted]),
                     ha='center', va='center', color='white' if normalised[true, predicted] > 0.5 else 'black')
    plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(num_classes), class_names)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title(title)
    plt.colorbar(label='Fraction of the true class')
    fig.tight_layout()
    return fig, confusion


# ---------------------------------------------------------------------------------------------------------------------
# Cross-validation of one experiment and sensor (section 11 of the notebook)
# ---------------------------------------------------------------------------------------------------------------------

def cross_validate(dataset_path, output_root, experiment, sensor, folds, keep_data):
    sensor_path = os.path.join(output_root, experiment, sensor)
    report = fold_report(parse_recordings(dataset_path, verbose=False), experiment, sensor)
    fold_results = []
    for fold in folds:
        start = time.time()
        fold_path = os.path.join(sensor_path, 'fold{}'.format(fold))
        if os.path.isdir(fold_path):
            remove_folder(fold_path)
        source_path = os.path.join(fold_path, 'data', 'source')
        prepared_path = os.path.join(fold_path, 'data', 'prepared')

        if report[fold]['missing_classes'] and report[fold]['validation']:
            log('    Warning: fold {} has no validation recordings of class {}'.format(
                fold, ', '.join(report[fold]['missing_classes'])))
        try:
            build_experiment(dataset_path, source_path, experiment, sensor, fold, verbose=False)
        except EmptyFoldError as error:
            log('  {} {} fold {}: SKIPPED - {}'.format(experiment, sensor, fold, error))
            result = dict(fold=fold, training_failed=False, skipped=True, reason=str(error))
            os.makedirs(fold_path, exist_ok=True)
            with open(os.path.join(fold_path, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)
            fold_results.append(result)
            continue
        prepare_dataset(source_path, prepared_path)
        train_npz, validation_npz, train_loader, validation_loader = make_loaders(prepared_path)
        class_names = list(train_npz['class_names'])
        title = '{} - {} - fold {}'.format(experiment, sensor, fold)

        try:
            net, history, best_epoch, seed = train_model(train_loader, validation_loader, len(class_names), prepared_path)
        except TrainingFailedError as error:
            log('  {} {} fold {}: TRAINING FAILED ({:.0f} s)\n    {}'.format(
                experiment, sensor, fold, time.time() - start, str(error).replace('\n', '\n    ')))
            if error.history is not None:
                save_figure(plot_training_curves(error.history, title + ' (training failed)'),
                            os.path.join(fold_path, 'training_curves_failed'))
            result = dict(fold=fold, training_failed=True, skipped=False, error=str(error))
            with open(os.path.join(fold_path, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)
            fold_results.append(result)
            if not keep_data:
                remove_folder(os.path.join(fold_path, 'data'))
            continue

        result, window_predictions = evaluate(net, validation_loader, validation_npz, prepared_path, best_epoch, seed)
        result.update(fold=fold, training_failed=False, skipped=False, history=history,
                      missing_validation_classes=report[fold]['missing_classes'])
        with open(os.path.join(fold_path, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        shutil.copy(os.path.join(prepared_path, 'snn_model.pt'), os.path.join(fold_path, 'snn_model.pt'))

        save_figure(plot_training_curves(history, title), os.path.join(fold_path, 'training_curves'))
        figure, _ = plot_confusion_matrix(validation_npz['labels'], window_predictions, class_names,
                                          'Confusion matrix (validation windows)\n' + title)
        save_figure(figure, os.path.join(fold_path, 'confusion_matrix'))

        fold_results.append(result)
        log('  {} {} fold {}: window accuracy {:.3f} | recording accuracy {:.3f} ({} recordings) | best epoch {} | seed {} | {:.0f} s'.format(
            experiment, sensor, fold, result['window_accuracy'], result['recording_accuracy'],
            result['num_validation_recordings'], best_epoch, seed, time.time() - start))
        if not keep_data:
            remove_folder(os.path.join(fold_path, 'data'))

    trained = [r for r in fold_results if not r['training_failed'] and not r.get('skipped')]
    window = np.array([r['window_accuracy'] for r in trained])
    recording = np.array([r['recording_accuracy'] for r in trained])
    summary = dict(experiment=experiment, sensor=sensor, description=EXPERIMENTS[experiment]['description'], folds=folds,
                   failed_folds=[r['fold'] for r in fold_results if r['training_failed']],
                   skipped_folds=[r['fold'] for r in fold_results if r.get('skipped')],
                   window_accuracy_mean=float(window.mean()) if trained else None,
                   window_accuracy_std=float(window.std()) if trained else None,
                   recording_accuracy_mean=float(recording.mean()) if trained else None,
                   recording_accuracy_std=float(recording.std()) if trained else None,
                   fold_results=[{k: v for k, v in r.items() if k != 'history'} for r in fold_results])
    with open(os.path.join(sensor_path, 'cv_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def save_architecture(output_root, experiment, class_names):
    """Saves the figure of the network of an experiment (it only depends on the number of classes and their names)."""
    torch.manual_seed(RANDOM_SEED)
    net = SpikingNet(NUM_ADDRESSES, NUM_HIDDEN, len(class_names), BETA)
    save_figure(plot_snn_architecture(net, class_names), os.path.join(output_root, experiment, 'snn_architecture'))


def write_summary(output_root):
    """Collects every cv_results.json below output_root into summary.json and summary.csv."""
    rows = []
    for path in sorted(glob.glob(os.path.join(output_root, '*', '*', 'cv_results.json'))):
        with open(path) as f:
            s = json.load(f)
        by_fold = {r['fold']: r for r in s['fold_results']}
        cross = by_fold.get(CROSS_SESSION_FOLD)
        row = dict(experiment=s['experiment'], sensor=s['sensor'],
                   window_accuracy_mean=s['window_accuracy_mean'], window_accuracy_std=s['window_accuracy_std'],
                   recording_accuracy_mean=s['recording_accuracy_mean'], recording_accuracy_std=s['recording_accuracy_std'],
                   failed_folds=' '.join(map(str, s['failed_folds'])),
                   skipped_folds=' '.join(map(str, s.get('skipped_folds', []))))
        done = lambda r: r is not None and not r['training_failed'] and not r.get('skipped')
        for fold in s['folds']:
            r = by_fold.get(fold)
            row['fold{}_window_accuracy'.format(fold)] = r['window_accuracy'] if done(r) else None
        row['cross_session_fold'] = CROSS_SESSION_FOLD if cross is not None else None
        row['cross_session_recording_accuracy'] = cross['recording_accuracy'] if done(cross) else None
        rows.append(row)

    with open(os.path.join(output_root, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(os.path.join(output_root, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows):
    log('\n{:<30} {:<6} {:>22} {:>25} {:>14} {:>13} {:>14}'.format(
        'Experiment', 'Sensor', 'Window accuracy', 'Recording accuracy', 'Fold {} window'.format(CROSS_SESSION_FOLD),
        'Failed folds', 'Skipped folds'))
    for r in rows:
        if r['window_accuracy_mean'] is None:
            accuracy = '{:>22} {:>25}'.format('no fold trained', '-')
        else:
            accuracy = '{:>14.3f} ± {:.3f} {:>17.3f} ± {:.3f}'.format(
                r['window_accuracy_mean'], r['window_accuracy_std'], r['recording_accuracy_mean'], r['recording_accuracy_std'])
        cross = r.get('fold{}_window_accuracy'.format(CROSS_SESSION_FOLD))
        log('{:<30} {:<6} {} {:>14} {:>13} {:>14}'.format(r['experiment'], r['sensor'], accuracy,
                                                          '-' if cross is None else '{:.3f}'.format(cross),
                                                          r['failed_folds'] or '-', r.get('skipped_folds') or '-'))


# ---------------------------------------------------------------------------------------------------------------------

def main():
    global _log_file, NUM_EPOCHS
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS), metavar='NAME',
                        help='experiments to run (default: all)')
    parser.add_argument('--sensors', nargs='+', default=SENSORS, choices=SENSORS, help='sensors to run (default: both)')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, NUM_FOLDS + 1)), metavar='K',
                        help='folds to run (default: all)')
    parser.add_argument('--dataset', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'Fan_NTAS_Dataset'),
                        help='folder with the Fan NTAS recordings and Class_Table.csv')
    parser.add_argument('--output', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'experiments', 'script_output'),
                        help='output folder')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='training epochs (default: {})'.format(NUM_EPOCHS))
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip the experiment/sensor pairs that already have a cv_results.json')
    parser.add_argument('--keep-data', action='store_true',
                        help='keep the recordings and windows of every fold (can take several GB)')
    parser.add_argument('--list', action='store_true', help='list the available experiments and exit')
    parser.add_argument('--check-folds', action='store_true',
                        help='show the recordings of every fold of the selected experiments and exit (no training)')
    args = parser.parse_args()

    if args.list:
        for name, definition in EXPERIMENTS.items():
            print('{:<28} {}'.format(name, definition['description']))
        return
    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        parser.error('unknown experiments: {}. Use --list to see the available ones.'.format(unknown))
    if any(fold not in range(1, NUM_FOLDS + 1) for fold in args.folds):
        parser.error('folds must be between 1 and {}'.format(NUM_FOLDS))
    if not os.path.isdir(args.dataset):
        parser.error('dataset folder not found: {}'.format(args.dataset))
    NUM_EPOCHS = args.epochs

    if args.check_folds:
        recordings = parse_recordings(args.dataset, verbose=True)
        print('\n{:<30} {:<6} {}'.format('Experiment', 'Sensor', ' | '.join('fold {} train/valid'.format(k) for k in args.folds)))
        for experiment in args.experiments:
            for sensor in args.sensors:
                report = fold_report(recordings, experiment, sensor)
                cells = ['{:>9}/{:<4}{}'.format(report[k]['train'], report[k]['validation'], ' ' if report[k]['complete'] else '*')
                         for k in args.folds]
                notes = ['fold {}: {}'.format(k, 'no recordings' if not report[k]['validation'] else
                                              'no ' + ', '.join(report[k]['missing_classes']))
                         for k in args.folds if not report[k]['complete']]
                print('{:<30} {:<6} {}  {}'.format(experiment, sensor, ' | '.join(cells), '; '.join(notes)))
        print('\n* = incomplete fold (no validation recordings, or classes without validation recordings)')
        return

    os.makedirs(args.output, exist_ok=True)
    _log_file = os.path.join(args.output, 'run_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    runs = [(e, s) for e in args.experiments for s in args.sensors]
    log('pyNTAVIS SNN cross-validation: {} experiment/sensor pairs, folds {}, {} epochs, device {}'.format(
        len(runs), args.folds, NUM_EPOCHS, device))
    log('Dataset: {}\nOutput:  {}'.format(args.dataset, args.output))

    start = time.time()
    for experiment, sensor in runs:
        if args.skip_existing and os.path.isfile(os.path.join(args.output, experiment, sensor, 'cv_results.json')):
            log('\n{} ({}): already done, skipped'.format(experiment, sensor))
            continue
        log('\n{} ({}): {}'.format(experiment, sensor, EXPERIMENTS[experiment]['description']))
        summary = cross_validate(args.dataset, args.output, experiment, sensor, args.folds, args.keep_data)

        # Figure of the network, once per experiment (the class names come from the first prepared fold)
        if not os.path.isfile(os.path.join(args.output, experiment, 'snn_architecture.png')):
            train, _ = select(parse_recordings(args.dataset, verbose=False), experiment, sensor, args.folds[0])
            class_names = sorted({c for _, c in train})     # same order as the class folders of prepare_dataset
            save_architecture(args.output, experiment, class_names)
        write_summary(args.output)
        log('  -> mean window accuracy: {}'.format(
            'no fold trained' if summary['window_accuracy_mean'] is None
            else '{:.3f} ± {:.3f}'.format(summary['window_accuracy_mean'], summary['window_accuracy_std'])))

    print_summary(write_summary(args.output))
    log('\nFinished in {:.0f} min. Results in {}'.format((time.time() - start) / 60, args.output))


if __name__ == '__main__':
    main()
