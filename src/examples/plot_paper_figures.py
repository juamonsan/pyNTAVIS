"""
Figures of the cross-validation results for the article, from the output of run_all_experiments.py.

Usage:
    python plot_paper_figures.py [--results snn_dataset/experiments/script_output]

Saves PDF (vector, for LaTeX) and PNG versions in <results>/paper_figures/:
    results_all_experiments    figure* (two columns): grouped bars, every experiment, both sensors
    results_dot_plot           one column: dot plot, one row per experiment, both sensors
    results_key                one column: grouped bars of the main experiments only
    results_all_vertical       one column: grouped bars of every experiment, group names rotated on the left

Bars and dots show the mean window accuracy over the 4 folds, with the standard deviation as error bars. Experiments
whose training did not converge in at least 2 folds are marked "n.l." (not learnable), as in the results table.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Experiment IDs as in the results table, grouped as in the experiments table
GROUPS = [
    ('Speed', [('1', '1_speed'), ('1b', '1b_speed_moving'), ('1c', '1c_speed_unseen_tape')]),
    ('Unbalance', [('2a min', '2a_unbalance_min'), ('2a mid', '2a_unbalance_mid'), ('2a max', '2a_unbalance_max'),
                   ('2a min+max', '2a_unbalance'), ('2a all', '2a_unbalance_all_speeds'), ('2b', '2b_unbalance_noise'),
                   ('2c', '2c_unbalance_unseen_noise'), ('2d min→max', '2d_unbalance_min_to_max'),
                   ('2d max→min', '2d_unbalance_max_to_min'), ('2e', '2e_unbalance_unseen_mid')]),
    ('Load', [('3 min', '3_load_min'), ('3 mid', '3_load_mid'), ('3 max', '3_load_max'), ('3 min+max', '3_load')]),
    ('State', [('4', '4_state')]),
    ('Tapping', [('5', '5_noise')]),
    ('Tape', [('6a', '6a_tape'), ('6b', '6b_tape_with_tapping'), ('6c', '6c_tape_same_day'), ('6d', '6d_tape_mid')]),
    ('Unbal. + tape', [('7a', '7a_unbalance_with_tape'), ('7b', '7b_unbalance_tape_mixed'), ('7c', '7c_unbalance_unseen_tape'),
                       ('7d', '7d_unbalance_mid_unseen_tape'), ('8', '8_unbalance_and_tape')]),
]
# Main experiments for the compact figure: (label, experiment)
KEY = [('Speed (1)', '1_speed'), ('Unbalance, min (2a)', '2a_unbalance_min'), ('Unbalance, mid (2a)', '2a_unbalance_mid'),
       ('Unbalance, max (2a)', '2a_unbalance_max'), ('Unbalance, unseen mid (2e)', '2e_unbalance_unseen_mid'),
       ('Load, max (3)', '3_load_max'), ('Machine state (4)', '4_state'), ('Tapping (5)', '5_noise'),
       ('Tape, same day (6c)', '6c_tape_same_day'), ('Tape, mid (6d)', '6d_tape_mid'),
       ('Unbalance, unseen tape (7c)', '7c_unbalance_unseen_tape')]

SENSORS = [('Accel', 'Accelerometer'), ('Mag', 'Magnetometer')]
# Categorical slots 1-2 of the reference palette (validated: CVD ΔE 24.7, normal-vision ΔE 33.6, contrast >= 3:1).
# Hatching (45° / 135°, darker tone of the same hue) keeps the sensors apart in grayscale print.
COLORS = {'Accel': '#2a78d6', 'Mag': '#eb6834'}
HATCH_COLORS = {'Accel': '#1c5cab', 'Mag': '#b84a1f'}
HATCHES = {'Accel': '////', 'Mag': '\\\\\\\\'}
INK, INK_2, INK_3, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
AXIS = '#000000'          # axes, ticks, tick labels, axis titles and group brackets
NOT_LEARNABLE_FOLDS = 2   # experiments with at least this many failed folds are "n.l."

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'xtick.color': AXIS, 'ytick.color': AXIS,
    'axes.labelcolor': AXIS, 'text.color': AXIS, 'hatch.linewidth': 0.5, 'pdf.fonttype': 42,
})


def load_results(results_path):
    """{experiment: {sensor: dict(mean, std, nl, chance)}} from the cv_results.json files."""
    data = {}
    experiments = [e for _, items in GROUPS for _, e in items]
    for e in experiments:
        data[e], num_classes = {}, None
        for sensor, _ in SENSORS:
            with open(os.path.join(results_path, e, sensor, 'cv_results.json')) as f:
                s = json.load(f)
            trained = [r for r in s['fold_results'] if not r['training_failed'] and not r.get('skipped')]
            if trained:
                num_classes = len(trained[0]['class_names'])
            data[e][sensor] = dict(mean=100 * (s['window_accuracy_mean'] or 0), std=100 * (s['window_accuracy_std'] or 0),
                                   nl=len(s['failed_folds']) >= NOT_LEARNABLE_FOLDS)
        for sensor, _ in SENSORS:
            data[e][sensor]['chance'] = 100 / num_classes
    return data


def style_axis(ax, value_axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    grid = ax.yaxis if value_axis == 'y' else ax.xaxis
    grid.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


def legend_handles(chance=True):
    handles = [Patch(facecolor=COLORS[s], edgecolor=HATCH_COLORS[s], hatch=HATCHES[s], linewidth=0, label=name)
               for s, name in SENSORS]
    if chance:
        handles.append(Line2D([], [], color=INK_2, linestyle=(0, (2, 1.5)), linewidth=0.8, label='Chance level'))
    return handles


def clipped_error(value, std):
    """Standard deviation as an error bar that stays inside 0-100 %, or None if there is no spread."""
    if std <= 0:
        return None
    return [[min(std, value)], [min(std, 100 - value)]]


def draw_bar(ax, x, value, std, sensor, width, horizontal=False):
    kwargs = dict(color=COLORS[sensor], edgecolor=HATCH_COLORS[sensor], hatch=HATCHES[sensor], linewidth=0)
    err = dict(ecolor=INK_2, elinewidth=0.6, capsize=1.2, capthick=0.6)
    if horizontal:
        ax.barh(x, value, height=width, xerr=clipped_error(value, std), error_kw=err, **kwargs)
    else:
        ax.bar(x, value, width=width, yerr=clipped_error(value, std), error_kw=err, **kwargs)


def figure_all(data, path):
    """A: every experiment, grouped bars, two-column width."""
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    width, gap = 0.38, 0.03            # bar width and the gap between the two bars of a group (as a fraction of a slot)
    x, ticks, labels, group_spans = 0.0, [], [], []
    for group, items in GROUPS:
        start = x
        for label, e in items:
            for k, (sensor, _) in enumerate(SENSORS):
                xb = x + (k - 0.5) * (width + gap)
                r = data[e][sensor]
                if r['nl']:
                    ax.text(xb, 3, 'n.l.', rotation=90, ha='center', va='bottom', fontsize=5.5, color=AXIS)
                else:
                    draw_bar(ax, xb, r['mean'], r['std'], sensor, width)
            chance = data[e]['Accel']['chance']
            ax.plot([x - 0.45, x + 0.45], [chance, chance], color=INK_2, linestyle=(0, (2, 1.5)), linewidth=0.8)
            ticks.append(x); labels.append(label); x += 1
        group_spans.append((group, start, x - 1)); x += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.7, x - 0.9); ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Window accuracy (%)')
    style_axis(ax)
    for group, a, b in group_spans:     # group names above the plot, with a thin bracket
        ax.annotate('', xy=(a - 0.4, 1.035), xytext=(b + 0.4, 1.035), xycoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-', color=AXIS, linewidth=0.6))
        ax.text((a + b) / 2, 1.06, group, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, color=AXIS)
    ax.legend(handles=legend_handles(), loc='lower center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=1.5)
    fig.tight_layout(pad=0.3)
    save(fig, path)


def figure_dots(data, path):
    """B: dot plot, one row per experiment, one column wide."""
    rows = [(label, e) for _, items in GROUPS for label, e in items]
    fig, ax = plt.subplots(figsize=(3.5, 4.6))
    y, yticks, ylabels, separators = 0, [], [], []
    for gi, (group, items) in enumerate(GROUPS):
        for label, e in items:
            values = {s: data[e][s] for s, _ in SENSORS}
            chance = values['Accel']['chance']
            ax.plot([chance, chance], [y - 0.35, y + 0.35], color=INK_2, linestyle=(0, (1.5, 1)), linewidth=0.7, zorder=1)
            for k, (sensor, _) in enumerate(SENSORS):
                v = values[sensor]
                yk = y + (k - 0.5) * 0.36       # each sensor slightly above / below, so equal values never hide each other
                if v['nl']:
                    ax.text(102, yk, 'n.l.', ha='left', va='center', fontsize=5.5, color=COLORS[sensor])
                    continue
                ax.errorbar(v['mean'], yk, xerr=clipped_error(v['mean'], v['std']), fmt='none', ecolor=COLORS[sensor],
                            elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=2)
                ax.plot(v['mean'], yk, marker='o' if sensor == 'Accel' else 's', markersize=4.0, color=COLORS[sensor],
                        markeredgecolor='white', markeredgewidth=0.6, zorder=3, linestyle='none')
            yticks.append(y); ylabels.append(label); y += 1
        if gi < len(GROUPS) - 1:
            separators.append(y - 0.5 + 0.25); y += 0.5
    for s in separators:
        ax.axhline(s, color=GRID, linewidth=0.6)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_xlim(0, 110); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Window accuracy (%)')
    style_axis(ax, value_axis='x')
    handles = [Line2D([], [], marker='o' if s == 'Accel' else 's', color=COLORS[s], markeredgecolor='white', linestyle='none',
                      markersize=5, label=name) for s, name in SENSORS]
    handles.append(Line2D([], [], color=INK_2, linestyle=(0, (1.5, 1)), linewidth=0.7, label='Chance level'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, handlelength=1.4,
              columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    save(fig, path)


def figure_key(data, path):
    """C: main experiments only, horizontal grouped bars, one column wide."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    height, gap = 0.36, 0.04
    for i, (label, e) in enumerate(KEY):
        for k, (sensor, _) in enumerate(SENSORS):
            yb = i + (k - 0.5) * (height + gap)
            v = data[e][sensor]
            if v['nl']:
                ax.text(2, yb, 'n.l.', ha='left', va='center', fontsize=5.5, color=AXIS)
            else:
                draw_bar(ax, yb, v['mean'], v['std'], sensor, height, horizontal=True)
        chance = data[e]['Accel']['chance']
        ax.plot([chance, chance], [i - 0.45, i + 0.45], color=INK_2, linestyle=(0, (2, 1.5)), linewidth=0.8)
    ax.set_yticks(range(len(KEY))); ax.set_yticklabels([label for label, _ in KEY]); ax.invert_yaxis()
    ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Window accuracy (%)')
    style_axis(ax, value_axis='x')
    ax.legend(handles=legend_handles(), loc='lower center', bbox_to_anchor=(0.35, 1.0), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    save(fig, path)


def figure_all_vertical(data, path):
    """D: every experiment, horizontal grouped bars, one column wide, group names rotated on the far left."""
    rows, spans, y = [], [], 0.0                # rows: (y, label, experiment); spans: (group, first y, last y)
    for gi, (group, items) in enumerate(GROUPS):
        if gi:
            y += 0.45                           # space between groups
        first = y
        for label, e in items:
            rows.append((y, label, e)); y += 1
        spans.append((group, first, y - 1))
    fig = plt.figure(figsize=(3.5, 4.4))
    ax = fig.add_axes([0.285, 0.09, 0.695, 0.85])    # [left, bottom, width, height]: room for the labels on the left
    height, gap = 0.36, 0.04
    for y, label, e in rows:
        for k, (sensor, _) in enumerate(SENSORS):
            yb = y + (k - 0.5) * (height + gap)
            v = data[e][sensor]
            if v['nl']:
                ax.text(2, yb, 'n.l.', ha='left', va='center', fontsize=6, color=AXIS)
            else:
                draw_bar(ax, yb, v['mean'], v['std'], sensor, height, horizontal=True)
        chance = data[e]['Accel']['chance']
        ax.plot([chance, chance], [y - 0.45, y + 0.45], color=INK_2, linestyle=(0, (2, 1.5)), linewidth=0.8)
    ax.set_yticks([y for y, _, _ in rows]); ax.set_yticklabels([label for _, label, _ in rows])
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    # Group names next to a thin bracket, placed just left of the widest experiment label of each group
    ax.set_ylim(rows[-1][0] + 0.7, rows[0][0] - 0.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    label_left = [to_axes.transform(t.get_window_extent(renderer))[0][0] for t in ax.get_yticklabels()]
    transform = ax.get_yaxis_transform()            # x in axes fraction, y in data units
    row_index = 0
    for group, first, last in spans:
        n = sum(1 for y, _, _ in rows if first <= y <= last)
        x = min(label_left[row_index:row_index + n]) - 0.02
        row_index += n
        ax.plot([x, x], [first - 0.4, last + 0.4], transform=transform, clip_on=False, color=AXIS, linewidth=0.6)
        if last > first:        # several rows: rotated name, centred on the bracket
            ax.text(x - 0.035, (first + last) / 2, group, transform=transform, rotation=90, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=AXIS, clip_on=False)
        else:                   # one row: the rotated name would be taller than the row, so write it horizontally
            ax.text(x - 0.02, first, group, transform=transform, ha='right', va='center',
                    fontsize=7, fontweight='bold', color=AXIS, clip_on=False)
    ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Window accuracy (%)', labelpad=2, fontsize=8)
    style_axis(ax, value_axis='x')
    ax.legend(handles=legend_handles(), loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False,
              handlelength=1.4, columnspacing=0.9, handletextpad=0.4, fontsize=7.5, bbox_transform=fig.transFigure)
    save(fig, path)


def save(fig, path):
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Figures of the cross-validation results for the article.')
    parser.add_argument('--results', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'experiments', 'script_output'))
    args = parser.parse_args()
    data = load_results(args.results)
    out = os.path.join(args.results, 'paper_figures')
    os.makedirs(out, exist_ok=True)
    figure_all(data, os.path.join(out, 'results_all_experiments'))
    figure_dots(data, os.path.join(out, 'results_dot_plot'))
    figure_key(data, os.path.join(out, 'results_key'))
    figure_all_vertical(data, os.path.join(out, 'results_all_vertical'))
    print('Figures saved in', out)


if __name__ == '__main__':
    main()
