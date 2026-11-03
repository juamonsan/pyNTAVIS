import matplotlib.pyplot as plt
from pyNTAVIS import *


def run(path, settings):
    three_axis_file = Loaders.loadAEDAT(path, settings)
    Functions.adapt_timestamps(three_axis_file, settings)

    Plots.spikegram(three_axis_file, settings)
    Plots.sonogram(three_axis_file, settings)
    Plots.histogram(three_axis_file, settings)
    x_activity, y_activity, z_activity, _ = Plots.average_activity(three_axis_file, settings)

    # Extract each axis as a mono SpikesFile (left_right = 0, 1, 2 for X, Y, Z)
    mono_settings = MainSettings(num_channels=settings.num_channels, mono_stereo=0, on_off_both=settings.on_off_both,
                                 address_size=settings.address_size, ts_tick=settings.ts_tick, bin_size=settings.bin_size)
    for axis, label in enumerate(settings.source_labels):
        axis_file = Functions.stereo_to_mono(three_axis_file, left_right=axis, settings=settings)
        Plots.spikegram(axis_file, mono_settings, graph_title='Spikegram - ' + label)
    plt.show()
