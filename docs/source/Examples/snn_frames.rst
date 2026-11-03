************************************
Prepare spiking information for SNNs
************************************

Spiking neural networks (SNNs) trained with time steps (for example with snnTorch, SpikingJelly or Norse) usually take
fixed-size inputs: a matrix of spike counts with one row per time step and one column per address. ``pyNTAVIS``
provides two functions to build these inputs from AER recordings of any kind (mono, stereo or three-axis).


1. Split a recording into windows
#################################

``Splitters.window_splitter()`` splits a SpikesFile into fixed-length time windows. The **stride** sets the time between
the start of two consecutive windows: by default it equals the window size (no overlap), and a smaller value gives
overlapping windows. Windows with fewer than **min_spikes** spikes are discarded, and only complete windows are
generated.

.. prompt:: python \

    from pyNTAVIS import *
    settings = MainSettings(num_channels=43, mono_stereo=MainSettings.THREE_AXIS, on_off_both=1, address_size=4, ts_tick=1)
    spikes_info = Loaders.loadAEDAT('src/examples/test_files/NTAS_Accel_3Axxis_43ch_ONOFF_addr4b_ts1.aedat', settings)
    Functions.adapt_timestamps(spikes_info, settings)

    # 200 ms windows with 50 % overlap; windows with fewer than 50 spikes are discarded
    windows = Splitters.window_splitter(spikes_info, settings, window_size=200000, stride=100000, min_spikes=50)

If **reset_timestamp** is True in the settings (the default), the timestamps of each window are relative to the start
of the window, so the position of every spike inside the window is preserved. Windows can also be saved to files with
**return_save_both** and **path**, like the other splitters.


2. Convert a window into frames
###############################

``Functions.spikes_to_frames()`` converts a SpikesFile into a matrix of spike counts with shape
``(num_steps, num_addresses)``, where ``num_addresses = settings.addresses_per_source * settings.num_sources``
(258 for the NTAS files above):

.. prompt:: python \

    import numpy as np

    # 1 ms time steps -> 200 x 258 matrix per window
    frames = np.stack([Functions.spikes_to_frames(w, settings, time_step=1000, duration=200000, start_ts=0) for w in windows])
    print(frames.shape)    # (number of windows, 200, 258)

Setting **duration** to the window size and **start_ts** to 0 gives every window the same shape. With
**binary=True**, each element only says whether the address fired in that time step (1) or not (0), instead of the
number of spikes.


3. Train an SNN
###############

The `SNN training notebook <https://github.com/juamonsan/pyNTAVIS/blob/master/src/examples/pyNTAVIS_SNN_training_snnTorch.ipynb>`_
uses these two functions to prepare train and validation sets from folders of AEDAT files (one subfolder per class),
and then trains and evaluates a feed-forward SNN of leaky integrate-and-fire neurons with snnTorch. It can be opened in
Google Colab, and also runs the 4-fold cross-validation of the :doc:`NTAS-Fan dataset <../NTAS_Fan_dataset>`
experiments. The scripts ``run_all_experiments.py`` (cross-validation of all the experiments, with figures) and
``run_baseline.py`` (a non-spiking baseline with the same windows and folds) in ``src/examples`` do the same from the
command line.

.. note::
    The SNN notebook and scripts need PyTorch and snnTorch: ``pip install torch snntorch``.
