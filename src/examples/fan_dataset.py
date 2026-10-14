"""
Builds train/validation folders from the Fan NTAS dataset for the experiments of pyNTAVIS_SNN_training_snnTorch.ipynb.

File names follow the convention <sample>_<speed>_<sensor>_<magnets>_<noise>.aedat, where:
    sample:  sample number of the recording.
    speed:   0 (stop), 1 (minimum) or 2 (maximum).
    sensor:  Accel (accelerometer) or Mag (magnetometer).
    magnets: 3 (base, balanced), 4 and 5 (unbalanced) or 6 (balanced).
    noise:   Yes or No (external vibration, tapping on the table).

Every combination of speed, sensor, magnets and noise was recorded 3 times. The repetition of each recording (1, 2 or 3)
is its position inside its combination when sorted by sample number. Experiments use 3-fold cross-validation over the
repetitions: fold k validates on repetition k and trains on the other two, so no recording is in both sets.
"""

import os
import shutil
from collections import defaultdict, namedtuple

Recording = namedtuple('Recording', ['path', 'sample', 'speed', 'sensor', 'magnets', 'noise', 'repetition'])

SENSORS = ['Accel', 'Mag']
NUM_REPETITIONS = 3
SPEED_NAMES = {0: 'stop', 1: 'min', 2: 'max'}


def balance(recording):
    return 'balanced' if recording.magnets in (3, 6) else 'unbalanced'


# Each experiment defines which recordings are used for training and validation and the class name of each recording.
# 'validation' defaults to 'train' when both sets use the same kind of recordings.
EXPERIMENTS = {
    '1_speed': dict(
        description='Speed (stop / min / max). All loads, no external noise.',
        train=lambda r: r.noise == 'No',
        label=lambda r: SPEED_NAMES[r.speed]),
    '2a_unbalance': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Min and max speed, no external noise.',
        train=lambda r: r.speed > 0 and r.noise == 'No',
        label=balance),
    '2a_unbalance_min': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Min speed only, no external noise.',
        train=lambda r: r.speed == 1 and r.noise == 'No',
        label=balance),
    '2a_unbalance_max': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Max speed only, no external noise.',
        train=lambda r: r.speed == 2 and r.noise == 'No',
        label=balance),
    '2b_unbalance_noise': dict(
        description='Balanced vs unbalanced. Min and max speed, with and without external noise in both sets.',
        train=lambda r: r.speed > 0,
        label=balance),
    '2c_unbalance_unseen_noise': dict(
        description='Balanced vs unbalanced. Train without external noise, validate only with external noise.',
        train=lambda r: r.speed > 0 and r.noise == 'No',
        validation=lambda r: r.speed > 0 and r.noise == 'Yes',
        label=balance),
    '2d_unbalance_min_to_max': dict(
        description='Balanced vs unbalanced, no external noise. Train at min speed, validate at max speed.',
        train=lambda r: r.speed == 1 and r.noise == 'No',
        validation=lambda r: r.speed == 2 and r.noise == 'No',
        label=balance),
    '2d_unbalance_max_to_min': dict(
        description='Balanced vs unbalanced, no external noise. Train at max speed, validate at min speed.',
        train=lambda r: r.speed == 2 and r.noise == 'No',
        validation=lambda r: r.speed == 1 and r.noise == 'No',
        label=balance),
    '3_load_min': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Min speed, no external noise.',
        train=lambda r: r.speed == 1 and r.noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '3_load_max': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Max speed, no external noise.',
        train=lambda r: r.speed == 2 and r.noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '3_load': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Min and max speed together, no external noise.',
        train=lambda r: r.speed > 0 and r.noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '4_state': dict(
        description='Machine state: stop / min-balanced / min-unbalanced / max-balanced / max-unbalanced. No external noise.',
        train=lambda r: r.noise == 'No',
        label=lambda r: 'stop' if r.speed == 0 else SPEED_NAMES[r.speed] + '_' + balance(r)),
    '5_noise': dict(
        description='External noise (tapping) vs no external noise. All speeds and loads.',
        train=lambda r: True,
        label=lambda r: 'tapping' if r.noise == 'Yes' else 'no_tapping'),
}


def parse_recordings(dataset_path, verbose=True):
    """
    Reads the metadata of every AEDAT file of the dataset from its name.

    Sensor names are matched without case and by their first letters, so small typos such as 'mag' or 'Magl' are accepted.

    Returns:
            Recording[]: One Recording per file, with its repetition number.
    """
    parsed = []
    for name in sorted(os.listdir(dataset_path)):
        if not name.lower().endswith('.aedat'):
            continue
        fields = os.path.splitext(name)[0].split('_')
        if len(fields) != 5:
            print('[fan_dataset] > Skipping file with unexpected name:', name)
            continue
        sample, speed, sensor, magnets, noise = fields
        sensor = 'Mag' if sensor.lower().startswith('mag') else 'Accel' if sensor.lower().startswith('acc') else None
        noise = noise.capitalize()
        if sensor is None or noise not in ('Yes', 'No') or speed not in ('0', '1', '2'):
            print('[fan_dataset] > Skipping file with unexpected name:', name)
            continue
        parsed.append((int(sample), int(speed), sensor, int(magnets), noise, os.path.join(dataset_path, name)))

    # Repetition = position inside its combination when sorted by sample number
    combinations = defaultdict(list)
    for sample, speed, sensor, magnets, noise, path in sorted(parsed):
        combinations[(speed, sensor, magnets, noise)].append((sample, path))

    recordings = []
    for (speed, sensor, magnets, noise), files in combinations.items():
        if len(files) != NUM_REPETITIONS and verbose:
            print('[fan_dataset] > Warning: {} recordings for speed {}, {}, {} magnets, noise {}'.format(
                len(files), speed, sensor, magnets, noise))
        for repetition, (sample, path) in enumerate(files, start=1):
            recordings.append(Recording(path, sample, speed, sensor, magnets, noise, repetition))

    if verbose:
        print('[fan_dataset] > {} recordings, {} combinations'.format(len(recordings), len(combinations)))
    return sorted(recordings, key=lambda r: r.sample)


def select(recordings, experiment, sensor, fold):
    """
    Returns the (recording, class name) pairs of the train and validation sets of an experiment, sensor and fold.
    """
    definition = EXPERIMENTS[experiment]
    validation_filter = definition.get('validation', definition['train'])
    train, validation = [], []
    for r in recordings:
        if r.sensor != sensor:
            continue
        if r.repetition != fold and definition['train'](r):
            train.append((r, definition['label'](r)))
        if r.repetition == fold and validation_filter(r):
            validation.append((r, definition['label'](r)))
    return train, validation


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_experiment(dataset_path, output_path, experiment, sensor, fold, verbose=True):
    """
    Creates output_path/train/<class>/ and output_path/validation/<class>/ with the recordings of an experiment.

    Files are hard-linked to the original recordings when the file system allows it (no extra disk space), and copied
    otherwise. output_path is emptied first, so it always matches the requested experiment.

    Parameters:
            dataset_path (string): Folder with the original AEDAT recordings.
            output_path (string): Folder where the train and validation folders are created.
            experiment (string): One of the keys of EXPERIMENTS.
            sensor (string): 'Accel' or 'Mag'.
            fold (int): Repetition used for validation (1, 2 or 3).
    """
    if experiment not in EXPERIMENTS:
        raise ValueError('Unknown experiment {}. Available: {}'.format(experiment, list(EXPERIMENTS)))
    if sensor not in SENSORS:
        raise ValueError('sensor should be one of {}'.format(SENSORS))
    if fold not in range(1, NUM_REPETITIONS + 1):
        raise ValueError('fold should be 1, 2 or 3')

    train, validation = select(parse_recordings(dataset_path, verbose=verbose), experiment, sensor, fold)

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    for split, pairs in [('train', train), ('validation', validation)]:
        for recording, class_name in pairs:
            class_path = os.path.join(output_path, split, class_name)
            os.makedirs(class_path, exist_ok=True)
            _link_or_copy(recording.path, os.path.join(class_path, os.path.basename(recording.path)))

    if verbose:
        print('[fan_dataset] > {} ({}, fold {}): {}'.format(experiment, sensor, fold, EXPERIMENTS[experiment]['description']))
        for split, pairs in [('train', train), ('validation', validation)]:
            counts = defaultdict(int)
            for _, class_name in pairs:
                counts[class_name] += 1
            print('  {:<10} '.format(split) + ', '.join('{}: {}'.format(c, counts[c]) for c in sorted(counts)))
    return train, validation
