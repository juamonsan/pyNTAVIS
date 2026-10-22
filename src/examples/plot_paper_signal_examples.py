"""
Compact figure for the article with the four views of a three-axis recording offered by pyNAVIS: spikegram, sonogram,
histogram and average activity (the same views as pyNTAVIS_3_axis_aedats_Tutorial.ipynb), sized for one IEEE column.

The data of every panel is computed with the local pyNAVIS (Loaders, Functions, Splitters and Plots with
return_data / returned arrays); only the layout is drawn here, because the pyNAVIS plot functions create one full-size
figure each.

Usage:
    python plot_paper_signal_examples.py [--recording PATH] [--start 1.0] [--duration 0.25] [--output DIR]

Default recording: 0331_3_Accel_5_No_Yes.aedat (accelerometer, max speed, 5 magnets, tape), the recording whose event rate
is closest to the median of the dataset (77 201 events/s). Saves signal_views.pdf and signal_views.png.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
import numpy as np

from pyNAVIS import Functions, Loaders, MainSettings, Plots, Splitters

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDING = os.path.join(SCRIPT_PATH, 'snn_dataset', 'Fan_NTAS_Dataset', '0331_3_Accel_5_No_Yes.aedat')

# Axis colors: green, violet, magenta from the reference palette, validated all-pairs (CVD ΔE >= 17.6). Magenta is below
# 3:1 contrast on white, so the axes are also named in the legend and directly on the address panels.
AXIS_COLORS = ['#008300', '#6250d6', '#e87ba4']
AXIS_NAMES = ['X', 'Y', 'Z']
# Sequential one-hue ramp (reference blue ramp) for the sonogram
SONOGRAM_CMAP = LinearSegmentedColormap.from_list('blue_ramp', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
AXIS = '#000000'
GRID = '#e6e5e1'


def style(ax, grid_axis=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    if grid_axis:
        (ax.yaxis if grid_axis == 'y' else ax.xaxis).grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def panel_label(ax, letter):
    ax.text(-0.02, 1.04, '({})'.format(letter), transform=ax.transAxes, ha='right', va='bottom', fontsize=7, fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description='Spikegram, sonogram, histogram and average activity of one recording.')
    parser.add_argument('--recording', default=DEFAULT_RECORDING)
    parser.add_argument('--start', type=float, default=1.0, help='start of the excerpt in s (default 1.0)')
    parser.add_argument('--duration', type=float, default=0.25, help='length of the excerpt in s (default 0.25)')
    parser.add_argument('--bin', type=float, default=5.0, help='bin width of the sonogram and average activity in ms (default 5)')
    parser.add_argument('--output', default=os.path.join(SCRIPT_PATH, 'snn_dataset', 'experiments', 'script_output', 'paper_figures'))
    args = parser.parse_args()

    # ---- Data, computed with pyNAVIS -----------------------------------------------------------------------------
    settings = MainSettings(num_channels=43, mono_stereo=MainSettings.THREE_AXIS, on_off_both=1, address_size=4,
                            ts_tick=1, bin_size=int(args.bin * 1000), verbose=False)
    recording = Loaders.loadAEDAT(args.recording, settings)
    Functions.adapt_timestamps(recording, settings)

    start_us, duration_us = int(args.start * 1e6), int(args.duration * 1e6)
    # Consecutive windows of --duration from time 0; the excerpt is the one that starts at --start
    windows = Splitters.window_splitter(recording, settings, duration_us, stride=duration_us, min_spikes=1)
    excerpt = windows[int(round(start_us / duration_us))]

    sonogram = Plots.sonogram(excerpt, settings, return_data=True)                    # [addresses, bins]
    histogram, figure = Plots.histogram(recording, settings); plt.close(figure)        # whole recording
    *activity, figure = Plots.average_activity(excerpt, settings); plt.close(figure)  # one array per axis
    plt.style.use('default')     # the pyNAVIS plot functions change the global style

    block = settings.addresses_per_source
    num_addresses = block * settings.num_sources
    addresses, timestamps = np.asarray(excerpt.addresses), np.asarray(excerpt.timestamps) / 1000   # ms
    axis_of = np.minimum(addresses // block, settings.num_sources - 1)

    # ---- Figure --------------------------------------------------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
        'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'xtick.color': AXIS, 'ytick.color': AXIS,
        'axes.labelcolor': AXIS, 'text.color': AXIS, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.5))
    (ax_spk, ax_son), (ax_hist, ax_act) = axes
    excerpt_ms = duration_us / 1000

    # (a) Spikegram: every event as a dot, colored by axis
    for k in range(settings.num_sources):
        m = axis_of == k
        ax_spk.scatter(timestamps[m], addresses[m], s=0.15, color=AXIS_COLORS[k], linewidths=0, rasterized=True)
    # (b) Sonogram: events per address and bin
    # Square-root color scale: most bins hold a few events and a few hold many, so a linear scale would leave it white
    image = ax_son.imshow(sonogram, aspect='auto', origin='lower', cmap=SONOGRAM_CMAP, interpolation='nearest',
                          norm=PowerNorm(gamma=0.5, vmin=0, vmax=sonogram.max()),
                          extent=[0, excerpt_ms, 0, num_addresses], rasterized=True)
    for ax in (ax_spk, ax_son):
        for k in range(1, settings.num_sources):
            ax.axhline(k * block, color=AXIS, linewidth=0.4, linestyle=(0, (2, 1.5)))
        for k, name in enumerate(AXIS_NAMES):     # direct labels of the axis blocks
            ax.text(1.015, (k + 0.5) * block, name, transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=6.5, fontweight='bold', color=AXIS_COLORS[k] if ax is ax_spk else AXIS)
        ax.set_xlim(0, excerpt_ms); ax.set_ylim(0, num_addresses)
        ax.set_yticks([0, block, 2 * block, num_addresses]); ax.set_xlabel('Time (ms)', labelpad=1)
        style(ax)
    ax_spk.set_ylabel('Address', labelpad=1)
    ax_son.set_yticklabels([])
    colorbar = fig.colorbar(image, ax=ax_son, fraction=0.08, pad=0.13, aspect=12)
    colorbar.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label('Events / bin', fontsize=6, labelpad=2)

    # (c) Histogram: events per address of each axis over the whole recording
    channel_address = np.arange(block)
    for k in range(settings.num_sources):
        counts = histogram[k * block:(k + 1) * block] / 1000
        ax_hist.plot(channel_address, counts, color=AXIS_COLORS[k], linewidth=0.8)
    ax_hist.set_xlim(0, block - 1); ax_hist.set_xticks([0, 20, 40, 60, 80])
    ax_hist.set_xlabel('Address (within the axis)', labelpad=1); ax_hist.set_ylabel('Events (×10³)', labelpad=1)
    ax_hist.set_ylim(bottom=0)
    style(ax_hist, 'y')

    # (d) Average activity: events of each axis per bin
    bin_centres = (np.arange(len(activity[0])) + 0.5) * args.bin
    for k in range(settings.num_sources):
        ax_act.plot(bin_centres, activity[k], color=AXIS_COLORS[k], linewidth=0.8)
    ax_act.set_xlim(0, excerpt_ms); ax_act.set_ylim(bottom=0)
    ax_act.set_xlabel('Time (ms)', labelpad=1); ax_act.set_ylabel('Events / bin', labelpad=1)
    style(ax_act, 'y')

    for ax, letter in zip((ax_spk, ax_son, ax_hist, ax_act), 'abcd'):
        panel_label(ax, letter)
    handles = [Line2D([], [], color=c, linewidth=1.6, label='{} axis'.format(n)) for c, n in zip(AXIS_COLORS, AXIS_NAMES)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.5, columnspacing=1.5, fontsize=7)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.8, rect=(0, 0, 1, 0.93))

    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, 'signal_views')
    fig.savefig(out + '.pdf', dpi=600)
    fig.savefig(out + '.png', dpi=300)
    print('Recording: {} | {} events in {:.2f} s ({:.0f} events/s) | excerpt {:.2f}-{:.2f} s: {} events'.format(
        os.path.basename(args.recording), len(recording.addresses), recording.max_ts / 1e6,
        len(recording.addresses) / (recording.max_ts / 1e6), args.start, args.start + args.duration, len(excerpt.addresses)))
    print('Saved', out + '.pdf / .png')


if __name__ == '__main__':
    main()
