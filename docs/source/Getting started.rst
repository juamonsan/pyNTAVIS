Getting started
===============


After :doc:`installing <Installing>` ``pyNTAVIS``, you can start using it by creating a new Python file and importing the package:

.. prompt:: python \

   from pyNTAVIS import *

With that, you will be able to use all the provided functionalities, which are specified in the :doc:`pyNTAVIS package section <pyNTAVIS>`.


A set of code examples can be found in the ``src/examples`` folder of the `GitHub repository <https://github.com/juamonsan/pyNTAVIS>`_
after downloading or cloning it. These examples will show you how to use most of ``pyNTAVIS`` functionalities.
In ``src/`` you will find a ``main.py`` file that can be used to run those examples.

You can take a closer look at some of these main functionalities in the :doc:`Examples section <Examples/examples>`.


Notebooks
---------

The ``src/examples`` folder also contains Jupyter notebooks, which can be run locally or opened in Google Colab:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Notebook
     - Colab
     - Description
   * - `Three-axis AEDAT tutorial <https://github.com/juamonsan/pyNTAVIS/blob/master/src/examples/pyNTAVIS_3_axis_aedats_Tutorial.ipynb>`_
     - `Open <https://colab.research.google.com/github/juamonsan/pyNTAVIS/blob/master/src/examples/pyNTAVIS_3_axis_aedats_Tutorial.ipynb>`__
     - Load an NTAS file, plot it and split it into X, Y and Z.
   * - `SNN training with snnTorch <https://github.com/juamonsan/pyNTAVIS/blob/master/src/examples/pyNTAVIS_SNN_training_snnTorch.ipynb>`_
     - `Open <https://colab.research.google.com/github/juamonsan/pyNTAVIS/blob/master/src/examples/pyNTAVIS_SNN_training_snnTorch.ipynb>`__
     - Window the recordings, build frames, train and evaluate an SNN, and run the 4-fold cross-validation of the :doc:`NTAS-Fan <NTAS_Fan_dataset>` experiments.
   * - `Original pyNAVIS tutorial <https://github.com/juamonsan/pyNTAVIS/blob/master/src/examples/pyNAVIS_Tutorial_WS.ipynb>`_
     -
     - NAS (mono and stereo) tutorial from pyNAVIS.


.. toctree::
   :maxdepth: 1
   :hidden:

   Examples/examples
