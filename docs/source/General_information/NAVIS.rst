.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">


*******************************
pyNTAVIS, pyNAVIS and NAVIS
*******************************

``pyNTAVIS`` stands for Neuromorphic Tri-axial and Auditory VISualizer. It is a Python package to analyze and process spiking information obtained from neuromorphic sensors: neuromorphic cochleae, such as the NAS, and the Neuromorphic Tri-Axial Sensor (NTAS), which converts the data of a MEMS accelerometer or magnetometer into spikes. It keeps all the functionalities of pyNAVIS, and adds support for three-axis files and tools to prepare spiking information for SNNs. All the plots, splitters and conversion functions accept three-axis files.

``pyNTAVIS`` is the latest step of a line of software tools developed by the same research group, described below.


Previous works
==============

pyNAVIS
-------

``pyNAVIS`` stands for python-based Neuromorphic Auditory Visualizer (NAVIS). It is the Python package from which ``pyNTAVIS`` was forked, focused on spiking information obtained from neuromorphic auditory sensors (mono and stereo cochleae). It was developed based on many requests from different users and research groups with which its authors have collaborated, as a Python version of NAVIS with many other functionalities added.

If you move from NAVIS to pyNAVIS (or pyNTAVIS), you will find the same whole set of functionalities. Moreover, some new useful functionalities suggested by neuromorphic researchers in different research internships, international conferences and workshops (CapoCaccia Neuromorphic Workshop and Telluride Neuromorphic Cognition Engineering Workshop) have already been implemented.

``pyNAVIS`` and ``pyNTAVIS`` do not have any GUI, which makes them lighter and faster. They are also very useful not only for being used as standalone software packages, but also to be imported within a bigger project and benefit from their functionalities. Due to the lack of a user interface, adding new functionalities to them is easier and faster than in NAVIS.

NAVIS
-----

NAVIS is a GUI-based software tool for neuromorphic researchers to analyze spiking information obtained from a neuromorphic cochlea. `NAVIS was also developed by the same research group <https://github.com/jpdominguez/NAVIS-Tool>`_, and it is the foundation of what pyNAVIS, and therefore pyNTAVIS, currently are. `It was published in Neurocomputing journal <https://www.sciencedirect.com/science/article/pii/S0925231216315624>`_ as an Original Software Publication.

It was built on Windows Presentation Foundation (WPF), which allows an interactive GUI for fast analysis of a recorded file from a neuromorphic cochlea. As a counterpart, NAVIS is only available for Windows OS users due to its GUI framework dependency.


Comparison
==========

The following table summarizes the comparison between the main functionalities that pyNTAVIS, pyNAVIS and NAVIS have:


+------------------------------------------------------+------------+------------+------------+
|                    Functionality                     |  pyNTAVIS  |  pyNAVIS   |   NAVIS    |
+=======================+==============================+============+============+============+
| Loaders               | \- Load AEDAT                |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Load CSV                  |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Load ZynqGrabber (iCub)   |  |check_|  |  |check_|  | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+
| Plots                 | \- Spikegram                 |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Sonogram                  |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Histogram                 |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Average activity          |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Difference between L/R    |  |check_|  |  |check_|  |  |check_|  |
+-----------------------+------------------------------+------------+------------+------------+
| Savers                | \- Save to AEDAT             |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Save to CSV               |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Save to TXT               |  |check_|  |  |check_|  | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+
| Splitters             | \- Manual Splitter           |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Automatic Splitter        |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Segmenter                 |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Window splitter           |  |check_|  | |uncheck_| | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+
| Generators            | \- Random addresses          |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Shift                     |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Sweep                     |  |check_|  |  |check_|  | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+
| Dataset generation    | \- Phaselock dataset         |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Sonogram dataset          |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Histogram dataset         |  |check_|  |  |check_|  | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+
| Other functionalities | \- Mono to stereo            |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Stereo to mono            |  |check_|  |  |check_|  |  |check_|  |
|                       +------------------------------+------------+------------+------------+
|                       | \- Phaselock                 |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Extract channels activity |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Check input file          |  |check_|  |  |check_|  | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Three-axis (NTAS) files   |  |check_|  | |uncheck_| | |uncheck_| |
|                       +------------------------------+------------+------------+------------+
|                       | \- Frames for SNNs           |  |check_|  | |uncheck_| | |uncheck_| |
+-----------------------+------------------------------+------------+------------+------------+

Apart from that, it is also important to mention a few aspects:

- pyNTAVIS and pyNAVIS are cross platform, while NAVIS only works in Windows.
- pyNTAVIS and pyNAVIS can be easily integrated with other tools. Just by importing the library in your script you can start using any of its functionalities. NAVIS cannot be integrated with other projects unless you dig deep in the code and merge it with another WPF application.
- pyNTAVIS and pyNAVIS do not have a GUI, while NAVIS does.
- It is way easier to implement new functionalities in pyNTAVIS and pyNAVIS than in NAVIS. Developers only need the basics of Python for that. In NAVIS, developers would need expertise on C#, LINQ and WPF and integrate the new function in the GUI.
- Users are moving from NAVIS towards pyNAVIS as the reference software for neuromorphic audio processing and analysis. pyNTAVIS extends it to three-axis (inertial and magnetic) neuromorphic sensors, and only pyNTAVIS can process NTAS files.
