#################################################################################
##                                                                             ##
##    Copyright C 2018  Juan P. Dominguez-Morales                              ##
##                                                                             ##
##    This file is part of pyNAVIS.                                            ##
##                                                                             ##
##    Modified in 2026 by Juan M. Montes-Sanchez for pyNTAVIS, a fork          ##
##    of pyNAVIS: package renamed to pyNTAVIS and three-axis (NTAS)            ##
##    support added.                                                           ##
##                                                                             ##
##    pyNAVIS is free software: you can redistribute it and/or modify          ##
##    it under the terms of the GNU General Public License as published by     ##
##    the Free Software Foundation, either version 3 of the License, or        ##
##    (at your option) any later version.                                      ##
##                                                                             ##
##    pyNAVIS is distributed in the hope that it will be useful,               ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of           ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the              ##
##    GNU General Public License for more details.                             ##
##                                                                             ##
##    You should have received a copy of the GNU General Public License        ##
##    along with pyNAVIS.  If not, see <http://www.gnu.org/licenses/>.         ##
##                                                                             ##
#################################################################################


import copy
# import random
import numpy as np
import time
import os
from bisect import bisect_left, bisect_right

from .functions import Functions
from .savers import Savers
from .utils import Utils
from .objects import SpikesFile

class Splitters:

    @staticmethod
    def manual_splitter(spikes_file, settings, init, end = -1, return_save_both = 0, output_format = '.aedat', path=None):
        """
        Extract a portion of the input SpikesFile file.

        Parameters:
                spikes_file (SpikesFile): Input file.
                init (int): First timestamp from which to start extracting. 
                end (int): Last timestamp from which to stop extracting. 
                settings (MainSettings): Configuration parameters for the input file.
                return_save_both (int, optional): Set it to 0 to return the resultant SpikesFile, to 1 to save the SpikesFile in the output path, and to 2 to do both.
                output_format (string, optional): Output format of the file. Currently supports '.aedat', '.csv', ".txt" and ".txt_rel". See the Savers class for more information.
                path (string, optional): Path where the output file will be saved. Format should not be specified. Not needed if return_save_both is set to 0.

        Returns:
                SpikesFile: SpikesFile containing the extracted portion of the input file. Returned only if return_save_both is either 0 or 2.
        """

        #THIS IS NOT NEEDED IF TS ARE SORTED
        aedat_addr_ts = list(zip(spikes_file.addresses, spikes_file.timestamps))
        aedat_addr_ts = sorted(aedat_addr_ts, key=Utils.getKey)
        spikes_file_new = copy.deepcopy(spikes_file)
        spikes_file_new = Utils.extract_addr_and_ts(aedat_addr_ts)

        if end == -1:
            end = np.max(spikes_file.timestamps)

        a =  bisect_left(spikes_file_new.timestamps, init)
        b =  bisect_right(spikes_file_new.timestamps, end)

        spikes_file_new.addresses = spikes_file_new.addresses[a:b]
        spikes_file_new.timestamps = spikes_file_new.timestamps[a:b]

        if settings.reset_timestamp == True:
            minimum_ts = np.min(spikes_file_new.timestamps)
            # Ensure timestamps are a numpy array and subtract minimum_ts vectorized
            spikes_file_new.timestamps = np.array(spikes_file_new.timestamps) - minimum_ts
        spikes_file_new.min_ts = np.min(spikes_file_new.timestamps)
        spikes_file_new.max_ts = np.max(spikes_file_new.timestamps)

        if return_save_both == 0:
            return spikes_file_new
        elif return_save_both == 1 or return_save_both == 2:
            Savers.save_as_any(spikes_file_new, path=path, output_format=output_format, settings=settings) 
            if return_save_both == 2:
                return spikes_file_new


    @staticmethod
    def window_splitter(spikes_file, settings, window_size, stride = None, min_spikes = 1, return_save_both = 0, output_format = '.aedat', path = None):
        """
        Splits a SpikesFile into fixed-length time windows. Works with mono, stereo and three-axis files.

        Parameters:
                spikes_file (SpikesFile): Input file.
                settings (MainSettings): Configuration parameters for the input file.
                window_size (int): Length of each window, in the same units as the timestamps (us after adapt_timestamps).
                stride (int, optional): Time between the start of two consecutive windows. Defaults to window_size (no overlap). Use a smaller value for overlapping windows.
                min_spikes (int, optional): Windows with fewer spikes than this value are discarded.
                return_save_both (int, optional): Set it to 0 to return the list of windows, to 1 to save each window in the output path, and to 2 to do both.
                output_format (string, optional): Output format of the files. Currently supports '.aedat', '.csv', ".txt" and ".txt_rel". See the Savers class for more information.
                path (string, optional): Base path where the windows will be saved. Each window is saved as path + '_' + window index. Format should not be specified. Not needed if return_save_both is set to 0.

        Returns:
                SpikesFile[]: List with one SpikesFile per window. Returned only if return_save_both is either 0 or 2.

        Note:
                Only complete windows are generated: a recording shorter than window_size produces no windows.

                If settings.reset_timestamp is True, the timestamps of each window are relative to the start of the window (not to its first spike),
                so the position of every spike inside the window is preserved.
        """
        if stride is None:
            stride = window_size

        addresses = np.asarray(spikes_file.addresses)
        timestamps = np.asarray(spikes_file.timestamps)

        # Windows are located with a binary search, which needs sorted timestamps
        if np.any(np.diff(timestamps) < 0):
            order = np.argsort(timestamps, kind='stable')
            addresses = addresses[order]
            timestamps = timestamps[order]

        windows = []
        if len(timestamps) > 0:
            window_starts = np.arange(timestamps[0], timestamps[-1] - window_size + 1, stride)
            for start in window_starts:
                a, b = np.searchsorted(timestamps, [start, start + window_size])
                if b - a < max(min_spikes, 1):
                    continue

                window_timestamps = timestamps[a:b] - start if settings.reset_timestamp else timestamps[a:b]
                windows.append(SpikesFile(addresses[a:b], window_timestamps))

        if return_save_both == 1 or return_save_both == 2:
            for i, window in enumerate(windows):
                Savers.save_as_any(window, path=path + '_' + str(i), output_format=output_format, settings=settings)
        if return_save_both == 0 or return_save_both == 2:
            return windows


    @staticmethod
    def segmenter_RT(spikes_file, noise_threshold, bin_width, return_save_both = 0, output_format = '.aedat', path=None, settings = None, verbose = False):
        """
        Removes background noise.

        A FIFO of size noise_threshold will be filling up with every spike. If the timestamp difference between the \
             first and the last spike in the FIFO is lower or equal than bin_width, then the most recent spike is saved.

        Parameters:
                spikes_file (SpikesFile): Input file.
                noise_threshold (int): Size of the FIFO. 
                bin_width (int): Time difference (in ms).
                return_save_both (int, optional): Set it to 0 to return the resultant SpikesFile, to 1 to save the SpikesFile in the output path, and to 2 to do both.
                output_format (string, optional): Output format of the file. Currently supports '.aedat', '.csv', ".txt" and ".txt_rel". See the Savers class for more information.
                path (string, optional): Path where the output file will be saved. Format should not be specified. Not needed if return_save_both is set to 0.
                settings (MainSettings, optional): Configuration parameters for the output file. Only needed when saving the output as an AEDAT file.
                verbose (boolean, optional): Set to True if you want the execution time of the function to be printed.

        Returns:
                SpikesFile: SpikesFile containing the processed input file. Returned only if return_save_both is either 0 or 2.
        """

        current_ts = 0  # Timestamp of the current spike that it's being processed.
        spikes_processed = 0    # Number of spikes processed. Used to avoid saving spikes before the FIFO has filled completely.
        buffer_ts = np.zeros(int(noise_threshold))  # FIFO.
        spikes_filtered = SpikesFile([], [])  # SpikesFile where the output will be saved.

        if verbose == True: start_time = time.time()

        for i in range(len(spikes_file.timestamps)):
            current_ts = spikes_file.timestamps[i]
            current_addr = spikes_file.addresses[i]

            buffer_ts = np.roll(buffer_ts, -1)  # Shifting the information of the FIFO.
            buffer_ts[-1] = current_ts

            spikes_processed +=1

            if ((current_ts - buffer_ts[0]) <= bin_width * 1000) and (spikes_processed >= noise_threshold):
                spikes_filtered.addresses.append(current_addr)
                spikes_filtered.timestamps.append(current_ts) 

        if verbose == True: print('SEGMENTER_RT CALCULATION', time.time() - start_time)

        if return_save_both == 0:
            return spikes_filtered
        elif return_save_both == 1 or return_save_both == 2:
            Savers.save_as_any(spikes_filtered, path=path, output_format=output_format) 
            if return_save_both == 2:
                return spikes_filtered


    @staticmethod
    def automatic_splitter(spikes_file, noise_threshold, bin_width, path, settings, output_format = '.aedat'):
        """
        Generate a set of files from the one received as input based on the silences in between words/sounds. This function uses segmenter_RT.
        NOTE: more tests needed to confirm that it works in any case.
        
        Parameters:
                spikes_file (SpikesFile): Input file.                
                noise_threshold (int): Size of the FIFO. 
                bin_width (int): Time difference (in ms).
                path (string): Path where the output files will be saved. Format should not be specified.
                settings (MainSettings, optional): Configuration parameters for the output file. Only needed when saving the output as an AEDAT file.
                output_format (string, optional): Output format of the file. Currently supports '.aedat', '.csv', ".txt" and ".txt_rel". See the Savers class for more information.                
                
        Returns:
                None.
        """
        number_of_splits_generated = 0
        index_previous_split = 0
        new_spikes_file = SpikesFile([], [])
        spikes_filtered = Splitters.segmenter_RT(spikes_file,noise_threshold, bin_width, return_save_both = 0)

        for i in range(1, len(spikes_filtered.timestamps)):
            if spikes_filtered.timestamps[i] - spikes_filtered.timestamps[i-1] >= bin_width:
                new_spikes_file.addresses = spikes_filtered.addresses[index_previous_split:i]
                new_spikes_file.timestamps = spikes_filtered.timestamps[index_previous_split:i]
                index_previous_split = i
                number_of_splits_generated += 1

                Savers.save_as_any(new_spikes_file, os.join(path, "split") + str(number_of_splits_generated), output_format, settings)