"""
Builds train/validation folders from the Fan NTAS dataset for the experiments of pyNTAVIS_SNN_training_snnTorch.ipynb.

File names follow the convention <sample>_<speed>_<sensor>_<magnets>_<external>_<internal>.aedat, where:
    sample:   sample number of the recording.
    speed:    0 (stop), 1 (minimum), 2 (medium) or 3 (maximum).
    sensor:   Accel (accelerometer) or Mag (magnetometer).
    magnets:  3 (base, balanced), 4 and 5 (unbalanced) or 6 (balanced).
    external: Yes or No (external vibration noise, tapping on the table).
    internal: Yes or No (internal noise, a piece of electrical tape touching the fan blades).
Names of the first dataset version, without the internal field, are read as internal = No.

The metadata of every recording is also listed in Class_Table.csv (same columns as Class_Table.xlsx: Sample number,
Fan Speed, Sensor, Magnets/Weight, External Noise, Internal Noise, Date). Fan Speed can be written as Stop (or 0), Min,
Mid (or Medium) or Max, or as the numeric code 0-3. When the table exists, every file name is
checked against it, and files that don't match are skipped with a warning. The table also gives the recording date.

The repetition of each recording is its position inside its combination (speed, sensor, magnets, external and internal
noise) when sorted by sample number. Experiments use NUM_FOLDS-fold cross-validation over the repetitions: fold k
validates on repetition k and trains on the others, so no recording is in both sets. Repetitions above NUM_FOLDS are
always used for training.

Recording days (October 2026): min, max and stop without tape have repetitions 1-3 on day 8, 4 on day 14 and 5-6 on days
19-20, so fold 4 validates on day 14, a session that is not used for training. The recordings with tape (min and max)
are from day 14 (repetitions 1-5) and day 19 (repetition 6). Medium speed was recorded on days 19, 20 and 21. If a
combination has fewer than NUM_FOLDS repetitions, its missing folds have no validation recordings (see fold_report).
"""

import csv
import os
import shutil
from collections import defaultdict, namedtuple

Recording = namedtuple('Recording', ['path', 'sample', 'speed', 'sensor', 'magnets', 'external_noise', 'internal_noise',
                                     'date', 'repetition'])

SENSORS = ['Accel', 'Mag']
NUM_FOLDS = 4
STOP, MIN, MEDIUM, MAX = 0, 1, 2, 3
SPEED_NAMES = {STOP: 'stop', MIN: 'min', MEDIUM: 'medium', MAX: 'max'}
TABLE_NAME = 'Class_Table.csv'
_TABLE_SPEEDS = {'stop': STOP, '0': STOP, 'min': MIN, '1': MIN, 'medium': MEDIUM, 'mid': MEDIUM, '2': MEDIUM, 'max': MAX, '3': MAX}


def balance(recording):
    return 'balanced' if recording.magnets in (3, 6) else 'unbalanced'


def no_internal(recording):
    return recording.internal_noise == 'No'


def tape(recording):
    return 'tape' if recording.internal_noise == 'Yes' else 'no_tape'


class EmptyFoldError(ValueError):
    """Raised by build_experiment when a fold has no training or no validation recordings."""


def repetition_fold(recording):
    """Default fold of a recording: its repetition. Repetitions above NUM_FOLDS are always used for training (None)."""
    return recording.repetition if recording.repetition <= NUM_FOLDS else None


# Each experiment defines which recordings are used for training and validation and the class name of each recording.
# 'validation' defaults to 'train' when both sets use the same kind of recordings.
# 'fold' gives the fold in which a recording is used for validation (None: always training). Defaults to repetition_fold.
EXPERIMENTS = {
    # Recording day
    '0_session_check': dict(
        description='Recording day: 8 October (last repetition) vs 14 October. Min and max speed, no external or internal noise. '
                    'Leave-one-load-out folds: fold k validates on the recordings with k + 2 magnets.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No' and r.repetition in (3, 4),
        label=lambda r: 'day_' + r.date,
        fold=lambda r: r.magnets - 2),

    # Speed
    '1_speed': dict(
        description='Speed (stop / min / medium / max, the speeds present in the dataset). All loads, no external or internal noise.',
        train=lambda r: no_internal(r) and r.external_noise == 'No',
        label=lambda r: SPEED_NAMES[r.speed]),
    '1b_speed_moving': dict(
        description='Speed of the moving fan (min / medium / max). All loads, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MEDIUM, MAX) and r.external_noise == 'No',
        label=lambda r: SPEED_NAMES[r.speed]),
    '1c_speed_unseen_tape': dict(
        description='Speed of the moving fan (min / medium / max), no external noise. Train without tape, validate only with '
                    'tape (the tape slows the fan by up to 6 %).',
        train=lambda r: no_internal(r) and r.speed in (MIN, MEDIUM, MAX) and r.external_noise == 'No',
        validation=lambda r: r.internal_noise == 'Yes' and r.speed in (MIN, MEDIUM, MAX) and r.external_noise == 'No',
        label=lambda r: SPEED_NAMES[r.speed]),

    # Unbalance: balanced {3, 6} vs unbalanced {4, 5}
    '2a_unbalance_min': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Min speed only, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MIN and r.external_noise == 'No',
        label=balance),
    '2a_unbalance_mid': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Medium speed only, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MEDIUM and r.external_noise == 'No',
        label=balance),
    '2a_unbalance_max': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Max speed only, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MAX and r.external_noise == 'No',
        label=balance),
    '2a_unbalance': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Min and max speed, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=balance),
    '2a_unbalance_all_speeds': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}. Min, medium and max speed together, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MEDIUM, MAX) and r.external_noise == 'No',
        label=balance),
    '2b_unbalance_noise': dict(
        description='Balanced vs unbalanced. Min and max speed, with and without external noise in both sets. No internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX),
        label=balance),
    '2c_unbalance_unseen_noise': dict(
        description='Balanced vs unbalanced. Train without external noise, validate only with external noise. No internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No',
        validation=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'Yes',
        label=balance),
    '2d_unbalance_min_to_max': dict(
        description='Balanced vs unbalanced, no external or internal noise. Train at min speed, validate at max speed.',
        train=lambda r: no_internal(r) and r.speed == MIN and r.external_noise == 'No',
        validation=lambda r: no_internal(r) and r.speed == MAX and r.external_noise == 'No',
        label=balance),
    '2d_unbalance_max_to_min': dict(
        description='Balanced vs unbalanced, no external or internal noise. Train at max speed, validate at min speed.',
        train=lambda r: no_internal(r) and r.speed == MAX and r.external_noise == 'No',
        validation=lambda r: no_internal(r) and r.speed == MIN and r.external_noise == 'No',
        label=balance),
    '2e_unbalance_unseen_mid': dict(
        description='Balanced vs unbalanced, no external or internal noise. Train at min and max speed, validate at medium '
                    'speed (a speed the network has never seen).',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No',
        validation=lambda r: no_internal(r) and r.speed == MEDIUM and r.external_noise == 'No',
        label=balance),

    # Load (number of magnets)
    '3_load_min': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Min speed, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MIN and r.external_noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '3_load_mid': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Medium speed, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MEDIUM and r.external_noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '3_load_max': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Max speed, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed == MAX and r.external_noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),
    '3_load': dict(
        description='Number of magnets (3 / 4 / 5 / 6). Min and max speed together, no external or internal noise.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=lambda r: str(r.magnets) + '_magnets'),

    # Machine state
    '4_state': dict(
        description='Machine state: stop / <speed>-balanced / <speed>-unbalanced for every speed present (min, medium, max). No external or internal noise.',
        train=lambda r: no_internal(r) and r.external_noise == 'No',
        label=lambda r: 'stop' if r.speed == STOP else SPEED_NAMES[r.speed] + '_' + balance(r)),

    # External noise (tapping on the table)
    '5_noise': dict(
        description='External noise (tapping) vs no external noise. All speeds and loads, no internal noise.',
        train=no_internal,
        label=lambda r: 'tapping' if r.external_noise == 'Yes' else 'no_tapping'),

    # Internal noise (tape touching the blades): min and max speed on days 14 and 19, mid speed on days 19-21.
    # At min and max speed, almost all tape recordings are from day 14: see 0_session_check and 6c_tape_same_day
    '6a_tape': dict(
        description='Internal noise (tape) vs no internal noise. Min and max speed, all loads, no external noise.',
        train=lambda r: r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=tape),
    '6b_tape_with_tapping': dict(
        description='Internal noise (tape) vs no internal noise. Min and max speed, all loads, with and without external noise in both sets.',
        train=lambda r: r.speed in (MIN, MAX),
        label=tape),
    '6c_tape_same_day': dict(
        description='Internal noise (tape) vs no internal noise without the recording day as a shortcut. Min and max speed, '
                    'no external noise. Leave-one-load-out folds: fold k validates on the 14 October recordings with k + 2 '
                    'magnets (one repetition with tape and one without), and trains on the other loads from both days.',
        train=lambda r: r.speed in (MIN, MAX) and r.external_noise == 'No',
        validation=lambda r: (r.speed in (MIN, MAX) and r.external_noise == 'No' and r.date == '20261014'
                              and (r.internal_noise == 'No' or r.repetition == 4)),
        label=tape,
        fold=lambda r: r.magnets - 2),
    '6d_tape_mid': dict(
        description='Internal noise (tape) vs no internal noise at medium speed, no external noise. Both classes were '
                    'recorded on the same days (19, 20 and 21 October), so the recording day is not a shortcut.',
        train=lambda r: r.speed == MEDIUM and r.external_noise == 'No',
        label=tape),

    # Unbalance with tape
    '7a_unbalance_with_tape': dict(
        description='Balanced {3, 6} vs unbalanced {4, 5}, all recordings with tape. Min and max speed, no external noise.',
        train=lambda r: r.internal_noise == 'Yes' and r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=balance),
    '7b_unbalance_tape_mixed': dict(
        description='Balanced vs unbalanced. Min and max speed, no external noise, with and without tape in both sets.',
        train=lambda r: r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=balance),
    '7c_unbalance_unseen_tape': dict(
        description='Balanced vs unbalanced. Min and max speed, no external noise. Train without tape, validate only with tape.',
        train=lambda r: no_internal(r) and r.speed in (MIN, MAX) and r.external_noise == 'No',
        validation=lambda r: r.internal_noise == 'Yes' and r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=balance),
    '7d_unbalance_mid_unseen_tape': dict(
        description='Balanced vs unbalanced at medium speed, no external noise. Train without tape, validate only with tape.',
        train=lambda r: no_internal(r) and r.speed == MEDIUM and r.external_noise == 'No',
        validation=lambda r: r.internal_noise == 'Yes' and r.speed == MEDIUM and r.external_noise == 'No',
        label=balance),

    # Unbalance and tape together
    '8_unbalance_and_tape': dict(
        description='Balanced / unbalanced and tape / no tape (4 classes). Min and max speed, no external noise.',
        train=lambda r: r.speed in (MIN, MAX) and r.external_noise == 'No',
        label=lambda r: balance(r) + '_' + tape(r)),
}


def _parse_name(name):
    """Returns (sample, speed, sensor, magnets, external_noise, internal_noise) from a file name, or None if it doesn't follow the convention."""
    fields = os.path.splitext(name)[0].split('_')
    if len(fields) == 5:
        fields.append('No')     # first dataset version: no internal noise field
    if len(fields) != 6:
        return None
    sample, speed, sensor, magnets, external, internal = fields
    if (not sample.isdigit() or speed not in ('0', '1', '2', '3') or sensor not in SENSORS
            or magnets not in ('3', '4', '5', '6') or external not in ('Yes', 'No') or internal not in ('Yes', 'No')):
        return None
    return int(sample), int(speed), sensor, int(magnets), external, internal


def read_table(dataset_path):
    """
    Reads Class_Table.csv from the dataset folder.

    Comma and semicolon separators are both accepted (Excel uses a semicolon when exporting CSV files with Spanish settings).

    Returns:
            dict: {sample number: (speed, sensor, magnets, external_noise, internal_noise, date)}, or None if there is no table.
    """
    path = os.path.join(dataset_path, TABLE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, newline='', encoding='utf-8-sig') as f:
        text = f.read()
    delimiter = ';' if text.splitlines()[0].count(';') > text.splitlines()[0].count(',') else ','

    table = {}
    for row in csv.DictReader(text.splitlines(), delimiter=delimiter):
        row = {key.strip(): (value or '').strip() for key, value in row.items() if key}
        if not row.get('Sample number') or not row.get('Sensor'):
            continue    # empty rows (placeholders for future recordings)
        table[int(float(row['Sample number']))] = (
            _TABLE_SPEEDS[row['Fan Speed'].lower()], row['Sensor'], int(float(row['Magnets/Weight'])),
            row['External Noise'], row['Internal Noise'], row.get('Date', ''))
    return table


def parse_recordings(dataset_path, verbose=True):
    """
    Reads the metadata of every AEDAT file of the dataset from its name, and checks it against Class_Table.csv if it exists.

    Returns:
            Recording[]: One Recording per valid file, with its repetition number, sorted by sample number.
    """
    table = read_table(dataset_path)
    if table is None and verbose:
        print('[fan_dataset] > Warning: {} not found, file names are not checked and dates are unknown'.format(TABLE_NAME))

    parsed, samples_with_file = [], set()
    for name in sorted(os.listdir(dataset_path)):
        if not name.lower().endswith('.aedat'):
            continue
        fields = _parse_name(name)
        if fields is None:
            print('[fan_dataset] > Skipping file with unexpected name:', name)
            continue
        sample, speed, sensor, magnets, external, internal = fields
        date = ''
        if table is not None:
            if sample not in table:
                print('[fan_dataset] > Skipping file not listed in {}: {}'.format(TABLE_NAME, name))
                continue
            if sample in samples_with_file:
                print('[fan_dataset] > Skipping file with a repeated sample number:', name)
                continue
            if table[sample][:5] != fields[1:]:
                print('[fan_dataset] > Skipping file whose name does not match {}: {} (table: {})'.format(
                    TABLE_NAME, name, table[sample][:5]))
                continue
            date = table[sample][5]
        samples_with_file.add(sample)
        parsed.append((sample, speed, sensor, magnets, external, internal, date, os.path.join(dataset_path, name)))

    if table is not None and verbose:
        missing = sorted(set(table) - samples_with_file)
        if missing:
            print('[fan_dataset] > Warning: {} samples of {} have no file: {}'.format(len(missing), TABLE_NAME, missing))

    # Repetition = position inside its combination when sorted by sample number
    combinations = defaultdict(list)
    for sample, speed, sensor, magnets, external, internal, date, path in sorted(parsed):
        combinations[(speed, sensor, magnets, external, internal)].append((sample, date, path))

    recordings = []
    for (speed, sensor, magnets, external, internal), files in combinations.items():
        if len(files) < NUM_FOLDS and verbose:
            print('[fan_dataset] > Warning: only {} recordings for speed {}, {}, {} magnets, external noise {}, internal noise {}'.format(
                len(files), speed, sensor, magnets, external, internal))
        for repetition, (sample, date, path) in enumerate(files, start=1):
            recordings.append(Recording(path, sample, speed, sensor, magnets, external, internal, date, repetition))

    if verbose:
        print('[fan_dataset] > {} recordings, {} combinations'.format(len(recordings), len(combinations)))
    return sorted(recordings, key=lambda r: r.sample)


def select(recordings, experiment, sensor, fold):
    """
    Returns the (recording, class name) pairs of the train and validation sets of an experiment, sensor and fold.

    Recordings without a fold (by default, those whose repetition is above NUM_FOLDS) are always in the train set.
    """
    definition = EXPERIMENTS[experiment]
    validation_filter = definition.get('validation', definition['train'])
    fold_of = definition.get('fold', repetition_fold)
    train, validation = [], []
    for r in recordings:
        if r.sensor != sensor:
            continue
        if fold_of(r) != fold and definition['train'](r):
            train.append((r, definition['label'](r)))
        if fold_of(r) == fold and validation_filter(r):
            validation.append((r, definition['label'](r)))
    return train, validation


def fold_report(recordings, experiment, sensor):
    """
    Returns, for every fold of an experiment and sensor, the number of training and validation recordings and the
    classes that have no validation recordings.

    Returns:
            dict: {fold: dict(train=int, validation=int, missing_classes=[str], complete=bool)}
    """
    report = {}
    for fold in range(1, NUM_FOLDS + 1):
        train, validation = select(recordings, experiment, sensor, fold)
        missing = sorted({c for _, c in train} - {c for _, c in validation})
        report[fold] = dict(train=len(train), validation=len(validation), missing_classes=missing,
                            complete=bool(train) and bool(validation) and not missing)
    return report


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
            dataset_path (string): Folder with the original AEDAT recordings (and Class_Table.csv).
            output_path (string): Folder where the train and validation folders are created.
            experiment (string): One of the keys of EXPERIMENTS.
            sensor (string): 'Accel' or 'Mag'.
            fold (int): Repetition used for validation (1 to NUM_FOLDS).
    """
    if experiment not in EXPERIMENTS:
        raise ValueError('Unknown experiment {}. Available: {}'.format(experiment, list(EXPERIMENTS)))
    if sensor not in SENSORS:
        raise ValueError('sensor should be one of {}'.format(SENSORS))
    if fold not in range(1, NUM_FOLDS + 1):
        raise ValueError('fold should be between 1 and {}'.format(NUM_FOLDS))

    train, validation = select(parse_recordings(dataset_path, verbose=verbose), experiment, sensor, fold)
    if not train or not validation:
        raise EmptyFoldError('{} ({}, fold {}) has {} training and {} validation recordings: this fold cannot be run with '
                             'the current dataset.'.format(experiment, sensor, fold, len(train), len(validation)))
    missing = sorted({c for _, c in train} - {c for _, c in validation})
    if missing:
        print('[fan_dataset] > Warning: {} ({}, fold {}) has no validation recordings of class {}'.format(
            experiment, sensor, fold, ', '.join(missing)))

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
            dates = sorted({r.date for r, _ in pairs if r.date})
            for _, class_name in pairs:
                counts[class_name] += 1
            print('  {:<10} '.format(split) + ', '.join('{}: {}'.format(c, counts[c]) for c in sorted(counts))
                  + ('  (dates: {})'.format(', '.join(dates)) if dates else ''))
    return train, validation
