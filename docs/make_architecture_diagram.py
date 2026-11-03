"""
Draws the software architecture of pyNTAVIS (docs/source/_static/pyNTAVIS_arch.png): one box per module with its classes
and their public methods or attributes, and the other pyNTAVIS modules each one imports. Items added in pyNTAVIS are marked
with "+", and items extended to three-axis files with "*".

Usage (from the repository root):
    python docs/make_architecture_diagram.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', '_static', 'pyNTAVIS_arch.png')

# module: [(class, [members])]; '+' = new in pyNTAVIS, '*' = extended to three-axis files
MODULES = {
    'dataset_gen': [('DatasetGenerators', ['generate_sonogram_dataset()', 'generate_phaselock_dataset()',
                                           'generate_histogram_dataset()'])],
    'report_functions': [('ReportFunctions', ['*PDF_report()'])],
    'generators': [('Generators', ['random_addrs()', 'shift()', 'sweep()'])],
    'plots': [('Plots', ['*spikegram()', '*sonogram()', '*histogram()', '*average_activity()', 'difference_between_LR()',
                         'mso_heatmap()', 'mso_spikegram()', 'mso_localization_plot()', 'mso_histogram()'])],
    'loaders': [('Loaders', ['loadAEDAT()', 'loadCSV()', 'loadZynqGrabberData()', 'loadAEDATLocalization()',
                             'loadCSVLocalization()'])],
    'splitters': [('Splitters', ['manual_splitter()', '+window_splitter()', 'automatic_splitter()', 'segmenter_RT()'])],
    'functions': [('Functions', ['adapt_timestamps()', 'check_SpikesFile()', 'check_LocalizationFile()',
                                 'order_SpikesFile()', '*stereo_to_mono()', 'mono_to_stereo()', 'phase_lock()',
                                 'extract_channels_activities()', 'ISI()', '+spikes_to_frames()'])],
    'savers': [('Savers', ['save_AEDAT()', 'save_CSV()', 'save_TXT()', 'save_TXT_relativeTS()', 'save_as_any()'])],
    'utils': [('Utils', ['extract_addr_and_ts()', 'execution_time()', 'getKey()', 'order_timestamps()', 'get_info()'])],
    'objects': [('SpikesFile', ['addresses: list', 'timestamps: list']),
                ('LocalizationFile', ['mso_* / lso_*: lists'])],
    'main_settings': [('MainSettings', ['num_channels: int', '*mono_stereo: int (0, 1, 2)', 'on_off_both: int',
                                        'address_size: int', 'timestamp_size: int', 'ts_tick: float', 'bin_size: int',
                                        'reset_timestamp: bool', '+MONO, STEREO, THREE_AXIS', '+num_sources',
                                        '+addresses_per_source', '+source_labels']),
                      ('LocalizationSettings', ['mso_* / lso_*: int'])],
}

# Imports between modules (from ... import), as dependent -> dependency
IMPORTS = {
    'dataset_gen': ['functions', 'loaders', 'plots', 'savers'],
    'report_functions': ['functions', 'loaders', 'objects', 'plots'],
    'generators': ['main_settings', 'objects', 'savers'],
    'plots': ['functions'],
    'loaders': ['functions', 'objects'],
    'splitters': ['functions', 'objects', 'savers', 'utils'],
    'functions': ['objects', 'savers', 'utils'],
    'savers': ['utils'],
    'utils': ['objects'],
}

ROWS = [['dataset_gen', 'report_functions', 'generators'],
        ['plots', 'loaders', 'splitters'],
        ['functions', 'savers', 'utils'],
        ['objects', 'main_settings']]

# Layout (data units = inches)
BOX_W, COL_GAP, ROW_GAP, LINE, TITLE_H, USES_H, CLASS_H, PAD = 3.3, 0.35, 0.35, 0.19, 0.34, 0.26, 0.30, 0.10
COLORS = dict(frame='#E8833A', module='#9BBB59', module_bg='#F4F8EC', cls='#4F81BD', cls_bg='#F2F2F2', new='#2E7D32',
              extended='#1F5FA8', text='#222222', uses='#5A6B3A')


def box_height(module):
    return TITLE_H + USES_H + sum(CLASS_H + LINE * len(members) + PAD for _, members in MODULES[module])


def main():
    row_heights = [max(box_height(m) for m in row) for row in ROWS]
    width = 3 * BOX_W + 2 * COL_GAP + 0.6
    height = sum(row_heights) + ROW_GAP * (len(ROWS) - 1) + 1.15
    fig = plt.figure(figsize=(width, height), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis('off')

    ax.add_patch(FancyBboxPatch((0.08, 0.08), width - 0.16, height - 0.16, boxstyle='round,pad=0,rounding_size=0.08',
                                fill=False, ec=COLORS['frame'], lw=2.5))
    ax.text(0.25, 0.38, 'pyNTAVIS', fontsize=17, fontweight='bold', color=COLORS['frame'], va='center')
    ax.text(width - 0.25, 0.38, '+ new in pyNTAVIS     * extended to three-axis files',
            fontsize=9.5, color=COLORS['text'], va='center', ha='right')

    y = 0.75
    for row, row_h in zip(ROWS, row_heights):
        x0 = 0.3 + (3 - len(row)) * (BOX_W + COL_GAP) / 2
        for i, module in enumerate(row):
            x, h = x0 + i * (BOX_W + COL_GAP), box_height(module)
            ax.add_patch(FancyBboxPatch((x, y), BOX_W, h, boxstyle='round,pad=0,rounding_size=0.05',
                                        fc=COLORS['module_bg'], ec=COLORS['module'], lw=1.6))
            ax.add_patch(FancyBboxPatch((x, y), BOX_W, TITLE_H, boxstyle='round,pad=0,rounding_size=0.05',
                                        fc=COLORS['module'], ec=COLORS['module'], lw=1.6))
            ax.text(x + 0.1, y + TITLE_H / 2, 'pyNTAVIS.' + module, fontsize=10.5, fontweight='bold',
                    color='white', va='center')
            uses = ', '.join(IMPORTS.get(module, [])) or '—'
            ax.text(x + 0.12, y + TITLE_H + USES_H / 2, 'uses: ' + uses, fontsize=8, style='italic',
                    color=COLORS['uses'], va='center')
            cy = y + TITLE_H + USES_H
            for cls, members in MODULES[module]:
                ch = CLASS_H + LINE * len(members) + 0.04
                ax.add_patch(FancyBboxPatch((x + 0.12, cy), BOX_W - 0.24, ch, boxstyle='square,pad=0',
                                            fc=COLORS['cls_bg'], ec=COLORS['cls'], lw=1))
                ax.add_patch(FancyBboxPatch((x + 0.12, cy), BOX_W - 0.24, CLASS_H, boxstyle='square,pad=0',
                                            fc=COLORS['cls'], ec=COLORS['cls'], lw=1))
                ax.text(x + 0.22, cy + CLASS_H / 2, cls, fontsize=9.5, fontweight='bold', color='white', va='center')
                for k, member in enumerate(members):
                    mark, name = (member[0], member[1:]) if member[0] in '+*' else ('', member)
                    color = {'+': COLORS['new'], '*': COLORS['extended']}.get(mark, COLORS['text'])
                    ax.text(x + 0.26, cy + CLASS_H + LINE * (k + 0.55), (mark or ' ') + ' ' + name, fontsize=8.5,
                            color=color, va='center', family='DejaVu Sans Mono',
                            fontweight='bold' if mark else 'normal')
                cy += ch + PAD
        y += row_h + ROW_GAP

    fig.savefig(OUTPUT, facecolor='white')
    print('Saved', OUTPUT)


if __name__ == '__main__':
    main()
