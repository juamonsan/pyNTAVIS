**********
Installing
**********

This section describes how to install and set up ``pyNTAVIS``.

First, some :ref:`requirements <requirements>` need to be installed. If you already have a running Python 3 environment,
you can skip this step and move to the :ref:`installation <user_install>`.




 .. _requirements:

Requirements
================

``pyNTAVIS`` runs under Python 3.10 or later, so first you need to install and set up a Python 3 environment.

.. note::
   Creating a Python environment is not necessary, but it is highly recommended. Environments act as a sand box where you can install and play around without spoiling the base Python installation.


To create an environment with the ``venv`` module of Python, open a terminal and run:

.. prompt:: bash $

   python3 -m venv .venv
   source .venv/bin/activate

On Windows, activate it with ``.venv\Scripts\activate`` instead. Conda environments (``conda create -n pyNTAVIS_env python=3.12``, then ``conda activate pyNTAVIS_env``) work as well.

After that, head over to the :ref:`installation <user_install>` instructions.







.. _user_install:

Installation
============

``pyNTAVIS`` is not distributed on PyPI: it is installed from its `GitHub repository <https://github.com/juamonsan/pyNTAVIS>`_. Clone or download the repository and install the package in editable mode, so that changes in the code are used directly:

.. prompt:: bash $

   git clone https://github.com/juamonsan/pyNTAVIS.git
   cd pyNTAVIS/src
   pip install -e .

This will automatically install the dependencies needed to use this package, which are ``numpy``, ``scipy`` and ``matplotlib``. The exact versions used during development are listed in ``src/requirements.txt``.

.. note::
   The package is named ``pyNTAVIS``, so it can be installed next to the original ``pyNAVIS`` from PyPI (the audio-only version, without three-axis support) without replacing it. Code written for pyNAVIS works with pyNTAVIS after replacing ``from pyNAVIS import`` with ``from pyNTAVIS import``.

.. note::
   To update pyNTAVIS, pull the latest changes of the repository (``git pull``). Since it is installed in editable mode, no reinstall is needed.

   To uninstall pyNTAVIS use the following command:

   .. prompt:: bash $

      pip uninstall pyNTAVIS


.. _snn_install:

Optional: SNN training
======================

The SNN training notebook and the NTAS-Fan experiment scripts also need PyTorch and snnTorch:

.. prompt:: bash $

   pip install torch snntorch


.. _colab_install:

Google Colab
============

The notebooks in ``src/examples`` (see :doc:`Getting started <Getting started>`) can be opened in Google Colab. Their first cell clones the repository and installs ``pyNTAVIS`` (and snnTorch, when needed) automatically.


.. note::
   Now you have everything set up. Take a look at the ``main.py`` file, the notebooks and the examples under the ``src/examples`` folder to familiarize with ``pyNTAVIS``.
   Also, don't forget to take a look at the :doc:`pyNTAVIS package section <pyNTAVIS>`, which has all the information about classes and methods.

.. warning::
   If you want to add new functionalities to ``pyNTAVIS``, please visit the :doc:`Contributing section <Contributing>`.
